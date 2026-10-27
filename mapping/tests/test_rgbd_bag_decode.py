import struct

import numpy as np
import pytest

from mapping.pc.go2_mapping_rgbd import RgbdDecodeError, decode_camera_info, decode_image, decode_tf_message


class _Writer:
    def __init__(self):
        self.data = bytearray(b"\x00\x01\x00\x00")
        self.origin = 4

    def align(self, size):
        self.data.extend(b"\0" * ((-(len(self.data) - self.origin)) % size))

    def pack(self, fmt, value, alignment):
        self.align(alignment)
        self.data.extend(struct.pack(fmt, value))

    def u8(self, value):
        self.pack("<B", value, 1)

    def u32(self, value):
        self.pack("<I", value, 4)

    def i32(self, value):
        self.pack("<i", value, 4)

    def f64(self, value):
        self.pack("<d", value, 8)

    def string(self, value):
        raw = value.encode("utf-8") + b"\0"
        self.u32(len(raw))
        self.data.extend(raw)

    def header(self, frame_id):
        self.i32(2)
        self.u32(3)
        self.string(frame_id)

    def bytes(self):
        return bytes(self.data)


def _image_payload(encoding, array, *, bigendian=False, padding=0):
    values = np.asarray(array)
    height, width = values.shape[:2]
    row_bytes = values[0].nbytes
    writer = _Writer()
    writer.header("camera_optical_frame")
    writer.u32(height)
    writer.u32(width)
    writer.string(encoding)
    writer.u8(int(bigendian))
    writer.u32(row_bytes + padding)
    raw_rows = []
    for row in values:
        if bigendian and values.dtype.itemsize > 1:
            visible = row.astype(values.dtype.newbyteorder(">"), copy=False).tobytes()
        else:
            visible = row.tobytes()
        raw_rows.append(visible + b"\xaa" * padding)
    raw = b"".join(raw_rows)
    writer.u32(len(raw))
    writer.data.extend(raw)
    return writer.bytes()


@pytest.mark.parametrize(
    ("encoding", "array"),
    [
        ("rgb8", np.arange(18, dtype=np.uint8).reshape(2, 3, 3)),
        ("bgr8", np.arange(18, dtype=np.uint8).reshape(2, 3, 3)),
        ("mono8", np.arange(6, dtype=np.uint8).reshape(2, 3)),
        ("16UC1", np.arange(6, dtype=np.uint16).reshape(2, 3)),
        ("32FC1", np.arange(6, dtype=np.float32).reshape(2, 3)),
    ],
)
def test_image_decoder_preserves_known_encodings_and_row_padding(encoding, array):
    decoded = decode_image(_image_payload(encoding, array, padding=5))

    assert decoded.header_timestamp_ns == 2_000_000_003
    assert decoded.frame_id == "camera_optical_frame"
    assert np.array_equal(decoded.image, array)
    assert decoded.image.flags.writeable is False


def test_image_decoder_converts_big_endian_depth_to_native_values():
    source = np.asarray([[1, 256], [4096, 65535]], dtype=np.uint16)

    decoded = decode_image(_image_payload("16UC1", source, bigendian=True, padding=2))

    assert np.array_equal(decoded.image, source)
    assert decoded.image.dtype.isnative


def _camera_info_payload():
    writer = _Writer()
    writer.header("camera_color_optical_frame")
    writer.u32(240)
    writer.u32(424)
    writer.string("plumb_bob")
    writer.u32(5)
    for value in [0.0] * 5:
        writer.f64(value)
    for value in [301.9424743652344, 0.0, 218.02423095703125, 0.0, 301.6651916503906, 126.14513397216797, 0.0, 0.0, 1.0]:
        writer.f64(value)
    for value in np.eye(3).reshape(-1):
        writer.f64(value)
    for value in [301.9424743652344, 0.0, 218.02423095703125, 0.0, 0.0, 301.6651916503906, 126.14513397216797, 0.0, 0.0, 0.0, 1.0, 0.0]:
        writer.f64(value)
    writer.u32(0)
    writer.u32(0)
    for value in [0, 0, 0, 0]:
        writer.u32(value)
    writer.u8(0)
    return writer.bytes()


def test_camera_info_keeps_full_rectification_contract():
    info = decode_camera_info(_camera_info_payload())

    assert info.frame_id == "camera_color_optical_frame"
    assert info.k.shape == (3, 3)
    assert info.k[0, 0] == pytest.approx(301.9424743652344)
    assert info.d.size == 5
    assert info.r.shape == (3, 3)
    assert info.p.shape == (3, 4)
    assert len(info.message_sha256) == 64


def test_corrupt_image_dimensions_and_unknown_encoding_fail_closed():
    payload = bytearray(_image_payload("mono8", np.ones((2, 3), dtype=np.uint8)))
    payload[-1:] = b""
    with pytest.raises(RgbdDecodeError):
        decode_image(bytes(payload))
    with pytest.raises(RgbdDecodeError, match="unsupported"):
        decode_image(_image_payload("yuv422", np.ones((2, 3), dtype=np.uint8)))


def test_tf_decoder_preserves_frame_chain_and_rejects_bad_quaternion():
    writer = _Writer()
    writer.u32(1)
    writer.header("camera_link")
    writer.string("camera_color_optical_frame")
    for value in [0.01, 0.02, 0.03]:
        writer.f64(value)
    for value in [0.0, 0.0, 0.0, 1.0]:
        writer.f64(value)

    transform = decode_tf_message(writer.bytes())[0]

    assert transform.parent_frame_id == "camera_link"
    assert transform.child_frame_id == "camera_color_optical_frame"
    assert transform.translation_xyz == pytest.approx((0.01, 0.02, 0.03))

    bad = _Writer()
    bad.u32(1)
    bad.header("a")
    bad.string("b")
    for value in [0.0, 0.0, 0.0]:
        bad.f64(value)
    for value in [0.0, 0.0, 0.0, 0.0]:
        bad.f64(value)
    with pytest.raises(RgbdDecodeError, match="quaternion"):
        decode_tf_message(bad.bytes())
