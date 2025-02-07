
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/voxel_generator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/voxel_generator.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n#second/protos/voxel_generator.proto\x12\rsecond.protos\"\xe7\x01\n\x0eVoxelGenerator\x12\x12\n\nvoxel_size\x18\x01 \x03(\x02\x12\x19\n\x11point_cloud_range\x18\x02 \x03(\x02\x12&\n\x1emax_number_of_points_per_voxel\x18\x03 \x01(\r\x12!\n\x19\x66ull_empty_part_with_mean\x18\x04 \x01(\x08\x12\x17\n\x0f\x62lock_filtering\x18\x05 \x01(\x08\x12\x14\n\x0c\x62lock_factor\x18\x06 \x01(\x03\x12\x12\n\nblock_size\x18\x07 \x01(\x03\x12\x18\n\x10height_threshold\x18\x08 \x01(\x02\x62\x06proto3')
)




_VOXELGENERATOR = _descriptor.Descriptor(
  name='VoxelGenerator',
  full_name='second.protos.VoxelGenerator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='voxel_size', full_name='second.protos.VoxelGenerator.voxel_size', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='point_cloud_range', full_name='second.protos.VoxelGenerator.point_cloud_range', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_number_of_points_per_voxel', full_name='second.protos.VoxelGenerator.max_number_of_points_per_voxel', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='full_empty_part_with_mean', full_name='second.protos.VoxelGenerator.full_empty_part_with_mean', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='block_filtering', full_name='second.protos.VoxelGenerator.block_filtering', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='block_factor', full_name='second.protos.VoxelGenerator.block_factor', index=5,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='block_size', full_name='second.protos.VoxelGenerator.block_size', index=6,
      number=7, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height_threshold', full_name='second.protos.VoxelGenerator.height_threshold', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=55,
  serialized_end=286,
)

DESCRIPTOR.message_types_by_name['VoxelGenerator'] = _VOXELGENERATOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

VoxelGenerator = _reflection.GeneratedProtocolMessageType('VoxelGenerator', (_message.Message,), dict(
  DESCRIPTOR = _VOXELGENERATOR,
  __module__ = 'second.protos.voxel_generator_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.VoxelGenerator)
  ))
_sym_db.RegisterMessage(VoxelGenerator)


# @@protoc_insertion_point(module_scope)