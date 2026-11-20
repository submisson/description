"""ROS-independent PointCloud2 decoding and rigid transforms.

All fields are read using field offsets, row padding and the message byte order.
The merger publishes a common x/y/z/intensity float32 layout for the CARLA stack.
"""
import math
import numpy as np

# sensor_msgs/PointField datatype values.
FIELD_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
OUTPUT_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])


def decode_xyzi(cloud):
    """Read even organized/padded/big-endian clouds into a dense float32 array."""
    if cloud.width == 0 or cloud.height == 0:
        return np.empty((0, 4), dtype=np.float32)
    names, formats, offsets = [], [], []
    byteorder = '>' if cloud.is_bigendian else '<'
    fields = {f.name: f for f in cloud.fields}
    if not all(name in fields for name in ('x', 'y', 'z')):
        raise ValueError('Point cloud has no x/y/z fields')
    for name in ('x', 'y', 'z', 'intensity'):
        field = fields.get(name)
        if field is None:
            continue
        if field.count != 1 or field.datatype not in FIELD_TYPES:
            raise ValueError(f'Unsupported scalar field: {name}')
        field_type = np.dtype(byteorder + FIELD_TYPES[field.datatype])
        if field.offset < 0 or field.offset + field_type.itemsize > cloud.point_step:
            raise ValueError(f'Field {name} is outside point_step')
        names.append(name); formats.append(field_type); offsets.append(field.offset)
    if cloud.row_step < cloud.width * cloud.point_step:
        raise ValueError('row_step is smaller than a row of points')
    required = (cloud.height - 1)*cloud.row_step + cloud.width*cloud.point_step
    if len(cloud.data) < required:
        raise ValueError('Truncated point cloud data')
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': cloud.point_step})
    points = np.ndarray((cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
                        strides=(cloud.row_step, cloud.point_step))
    xyzi = np.zeros((cloud.width*cloud.height, 4), dtype=np.float32)
    for i, name in enumerate(('x', 'y', 'z', 'intensity')):
        if name in names:
            xyzi[:, i] = points[name].ravel()
    # Keep invalid intensity from propagating; only invalid coordinates drop points.
    xyzi[:, 3] = np.nan_to_num(xyzi[:, 3], nan=0, posinf=0, neginf=0)
    return xyzi[np.isfinite(xyzi[:, :3]).all(axis=1)]


def rotation_from_quaternion(x, y, z, w):
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid transform quaternion')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)


def transform_xyzi(points, rotation, translation):
    result = points.copy()
    result[:, :3] = points[:, :3] @ np.asarray(rotation).T + np.asarray(translation)
    return result


def crop_xyzi(points, boxes):
    """Remove points inside either vehicle crop box, after transforming to base_link."""
    keep = np.ones(len(points), dtype=bool)
    for box in boxes:
        inside = np.ones(len(points), dtype=bool)
        for index, axis in enumerate(('x', 'y', 'z')):
            inside &= (points[:, index] >= box['min_' + axis]) & (points[:, index] <= box['max_' + axis])
        keep &= ~inside
    return points[keep]


def encode_xyzi(points):
    # Explicit little-endian float32; output fields occupy 0,4,8,12 with point_step=16.
    return np.ascontiguousarray(points, dtype='<f4').tobytes()


def take_synchronized(buffers, stamp, tolerance_ns):
    """Consume one all-sensor group; never reuse frames or mix distant timestamps.

    Each deque is kept in nondecreasing stamp order by the caller. Oldest frames
    outside the synchronization window are discarded until all fronts match.
    """
    while all(buffers):
        times = [stamp(buffer[0]) for buffer in buffers]
        latest = max(times)
        if latest - min(times) <= tolerance_ns:
            return [buffer.popleft() for buffer in buffers]
        for buffer, time in zip(buffers, times):
            if latest - time > tolerance_ns:
                buffer.popleft()
    return None
