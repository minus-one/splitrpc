// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: infer.proto

#include "infer.pb.h"
#ifdef PROFILE_MODE
#include "time_utils.h"
#include "stats_utils.h"
#include <stdio.h>
uint64_t pbStartNs;
std::vector<uint64_t> pb_reqParse;
std::vector<uint64_t> pb_respParse;
std::vector<uint64_t> pb_reqSerialize;
std::vector<uint64_t> pb_respSerialize;
#endif

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

#ifdef PROFILE_MODE
void printPbStats() 
{
  printf("pb_reqParse Exec time stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      pb_reqParse.size(), getMean(pb_reqParse), getPercentile(pb_reqParse, 0.90), 
      getPercentile(pb_reqParse, 0.95), getPercentile(pb_reqParse, 0.99)); 
  printf("pb_respParse Exec time stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      pb_respParse.size(), getMean(pb_respParse), getPercentile(pb_respParse, 0.90), 
      getPercentile(pb_respParse, 0.95), getPercentile(pb_respParse, 0.99)); 
  printf("pb_reqSerialize Exec time stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      pb_reqSerialize.size(), getMean(pb_reqSerialize), getPercentile(pb_reqSerialize, 0.90), 
      getPercentile(pb_reqSerialize, 0.95), getPercentile(pb_reqSerialize, 0.99)); 
  printf("pb_respSerialize Exec time stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      pb_respSerialize.size(), getMean(pb_respSerialize), getPercentile(pb_respSerialize, 0.90), 
      getPercentile(pb_respSerialize, 0.95), getPercentile(pb_respSerialize, 0.99)); 
}
#endif

PROTOBUF_PRAGMA_INIT_SEG
namespace doinference {
constexpr ScheduleRequest::ScheduleRequest(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : payload_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , guid_(uint64_t{0u})
  , funcid_(0u)
  , size_(0u){}
struct ScheduleRequestDefaultTypeInternal {
  constexpr ScheduleRequestDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ScheduleRequestDefaultTypeInternal() {}
  union {
    ScheduleRequest _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ScheduleRequestDefaultTypeInternal _ScheduleRequest_default_instance_;
constexpr ScheduleReply::ScheduleReply(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : payload_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , guid_(uint64_t{0u})
  , funcid_(0u)
  , size_(0u){}
struct ScheduleReplyDefaultTypeInternal {
  constexpr ScheduleReplyDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ScheduleReplyDefaultTypeInternal() {}
  union {
    ScheduleReply _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ScheduleReplyDefaultTypeInternal _ScheduleReply_default_instance_;
}  // namespace doinference
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_infer_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_infer_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_infer_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_infer_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleRequest, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleRequest, guid_),
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleRequest, funcid_),
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleRequest, payload_),
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleRequest, size_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleReply, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleReply, guid_),
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleReply, funcid_),
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleReply, payload_),
  PROTOBUF_FIELD_OFFSET(::doinference::ScheduleReply, size_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::doinference::ScheduleRequest)},
  { 10, -1, -1, sizeof(::doinference::ScheduleReply)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::doinference::_ScheduleRequest_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::doinference::_ScheduleReply_default_instance_),
};

const char descriptor_table_protodef_infer_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\013infer.proto\022\013doinference\"N\n\017ScheduleRe"
  "quest\022\014\n\004guid\030\001 \001(\004\022\016\n\006funcid\030\002 \001(\r\022\017\n\007p"
  "ayload\030\003 \001(\014\022\014\n\004size\030\004 \001(\r\"L\n\rScheduleRe"
  "ply\022\014\n\004guid\030\001 \001(\004\022\016\n\006funcid\030\002 \001(\r\022\017\n\007pay"
  "load\030\003 \001(\014\022\014\n\004size\030\004 \001(\r2Q\n\tScheduler\022D\n"
  "\006runJob\022\034.doinference.ScheduleRequest\032\032."
  "doinference.ScheduleReply\"\000b\006proto3"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_infer_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_infer_2eproto = {
  false, false, 275, descriptor_table_protodef_infer_2eproto, "infer.proto", 
  &descriptor_table_infer_2eproto_once, nullptr, 0, 2,
  schemas, file_default_instances, TableStruct_infer_2eproto::offsets,
  file_level_metadata_infer_2eproto, file_level_enum_descriptors_infer_2eproto, file_level_service_descriptors_infer_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_infer_2eproto_getter() {
  return &descriptor_table_infer_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_infer_2eproto(&descriptor_table_infer_2eproto);
namespace doinference {

// ===================================================================

class ScheduleRequest::_Internal {
 public:
};

ScheduleRequest::ScheduleRequest(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:doinference.ScheduleRequest)
}
ScheduleRequest::ScheduleRequest(const ScheduleRequest& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  payload_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_payload().empty()) {
    payload_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_payload(), 
      GetArenaForAllocation());
  }
  ::memcpy(&guid_, &from.guid_,
    static_cast<size_t>(reinterpret_cast<char*>(&size_) -
    reinterpret_cast<char*>(&guid_)) + sizeof(size_));
  // @@protoc_insertion_point(copy_constructor:doinference.ScheduleRequest)
}

void ScheduleRequest::SharedCtor() {
payload_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&guid_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&size_) -
    reinterpret_cast<char*>(&guid_)) + sizeof(size_));
}

ScheduleRequest::~ScheduleRequest() {
  // @@protoc_insertion_point(destructor:doinference.ScheduleRequest)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void ScheduleRequest::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  payload_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void ScheduleRequest::ArenaDtor(void* object) {
  ScheduleRequest* _this = reinterpret_cast< ScheduleRequest* >(object);
  (void)_this;
}
void ScheduleRequest::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ScheduleRequest::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void ScheduleRequest::Clear() {
// @@protoc_insertion_point(message_clear_start:doinference.ScheduleRequest)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  payload_.ClearToEmpty();
  ::memset(&guid_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&size_) -
      reinterpret_cast<char*>(&guid_)) + sizeof(size_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ScheduleRequest::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#ifdef PROFILE_MODE
  pbStartNs = getCurNs();
#endif
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // uint64 guid = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          guid_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // uint32 funcid = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          funcid_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // bytes payload = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          auto str = _internal_mutable_payload();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // uint32 size = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
#ifdef PROFILE_MODE
  pb_reqParse.push_back(getCurNs() - pbStartNs);
#endif
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* ScheduleRequest::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
#ifdef PROFILE_MODE
  pbStartNs = getCurNs();
#endif
  // @@protoc_insertion_point(serialize_to_array_start:doinference.ScheduleRequest)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 guid = 1;
  if (this->_internal_guid() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(1, this->_internal_guid(), target);
  }

  // uint32 funcid = 2;
  if (this->_internal_funcid() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(2, this->_internal_funcid(), target);
  }

  // bytes payload = 3;
  if (!this->_internal_payload().empty()) {
    target = stream->WriteBytesMaybeAliased(
        3, this->_internal_payload(), target);
  }

  // uint32 size = 4;
  if (this->_internal_size() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(4, this->_internal_size(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:doinference.ScheduleRequest)

#ifdef PROFILE_MODE
  pb_reqSerialize.push_back(getCurNs() - pbStartNs);
#endif
  return target;
}

size_t ScheduleRequest::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:doinference.ScheduleRequest)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // bytes payload = 3;
  if (!this->_internal_payload().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::BytesSize(
        this->_internal_payload());
  }

  // uint64 guid = 1;
  if (this->_internal_guid() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64SizePlusOne(this->_internal_guid());
  }

  // uint32 funcid = 2;
  if (this->_internal_funcid() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32SizePlusOne(this->_internal_funcid());
  }

  // uint32 size = 4;
  if (this->_internal_size() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32SizePlusOne(this->_internal_size());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ScheduleRequest::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    ScheduleRequest::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ScheduleRequest::GetClassData() const { return &_class_data_; }

void ScheduleRequest::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<ScheduleRequest *>(to)->MergeFrom(
      static_cast<const ScheduleRequest &>(from));
}


void ScheduleRequest::MergeFrom(const ScheduleRequest& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:doinference.ScheduleRequest)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_payload().empty()) {
    _internal_set_payload(from._internal_payload());
  }
  if (from._internal_guid() != 0) {
    _internal_set_guid(from._internal_guid());
  }
  if (from._internal_funcid() != 0) {
    _internal_set_funcid(from._internal_funcid());
  }
  if (from._internal_size() != 0) {
    _internal_set_size(from._internal_size());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ScheduleRequest::CopyFrom(const ScheduleRequest& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:doinference.ScheduleRequest)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ScheduleRequest::IsInitialized() const {
  return true;
}

void ScheduleRequest::InternalSwap(ScheduleRequest* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &payload_, lhs_arena,
      &other->payload_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(ScheduleRequest, size_)
      + sizeof(ScheduleRequest::size_)
      - PROTOBUF_FIELD_OFFSET(ScheduleRequest, guid_)>(
          reinterpret_cast<char*>(&guid_),
          reinterpret_cast<char*>(&other->guid_));
}

::PROTOBUF_NAMESPACE_ID::Metadata ScheduleRequest::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_infer_2eproto_getter, &descriptor_table_infer_2eproto_once,
      file_level_metadata_infer_2eproto[0]);
}

// ===================================================================

class ScheduleReply::_Internal {
 public:
};

ScheduleReply::ScheduleReply(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:doinference.ScheduleReply)
}
ScheduleReply::ScheduleReply(const ScheduleReply& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  payload_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_payload().empty()) {
    payload_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_payload(), 
      GetArenaForAllocation());
  }
  ::memcpy(&guid_, &from.guid_,
    static_cast<size_t>(reinterpret_cast<char*>(&size_) -
    reinterpret_cast<char*>(&guid_)) + sizeof(size_));
  // @@protoc_insertion_point(copy_constructor:doinference.ScheduleReply)
}

void ScheduleReply::SharedCtor() {
payload_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&guid_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&size_) -
    reinterpret_cast<char*>(&guid_)) + sizeof(size_));
}

ScheduleReply::~ScheduleReply() {
  // @@protoc_insertion_point(destructor:doinference.ScheduleReply)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void ScheduleReply::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  payload_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void ScheduleReply::ArenaDtor(void* object) {
  ScheduleReply* _this = reinterpret_cast< ScheduleReply* >(object);
  (void)_this;
}
void ScheduleReply::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ScheduleReply::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void ScheduleReply::Clear() {
// @@protoc_insertion_point(message_clear_start:doinference.ScheduleReply)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  payload_.ClearToEmpty();
  ::memset(&guid_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&size_) -
      reinterpret_cast<char*>(&guid_)) + sizeof(size_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ScheduleReply::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#ifdef PROFILE_MODE
  pbStartNs = getCurNs();
#endif
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // uint64 guid = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          guid_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // uint32 funcid = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          funcid_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // bytes payload = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          auto str = _internal_mutable_payload();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // uint32 size = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:

#ifdef PROFILE_MODE
  pb_respParse.push_back(getCurNs() - pbStartNs);
#endif
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* ScheduleReply::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
#ifdef PROFILE_MODE
  pbStartNs = getCurNs();
#endif
  // @@protoc_insertion_point(serialize_to_array_start:doinference.ScheduleReply)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 guid = 1;
  if (this->_internal_guid() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(1, this->_internal_guid(), target);
  }

  // uint32 funcid = 2;
  if (this->_internal_funcid() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(2, this->_internal_funcid(), target);
  }

  // bytes payload = 3;
  if (!this->_internal_payload().empty()) {
    target = stream->WriteBytesMaybeAliased(
        3, this->_internal_payload(), target);
  }

  // uint32 size = 4;
  if (this->_internal_size() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(4, this->_internal_size(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:doinference.ScheduleReply)
#ifdef PROFILE_MODE
  pb_respSerialize.push_back(getCurNs() - pbStartNs);
#endif
  return target;
}

size_t ScheduleReply::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:doinference.ScheduleReply)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // bytes payload = 3;
  if (!this->_internal_payload().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::BytesSize(
        this->_internal_payload());
  }

  // uint64 guid = 1;
  if (this->_internal_guid() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64SizePlusOne(this->_internal_guid());
  }

  // uint32 funcid = 2;
  if (this->_internal_funcid() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32SizePlusOne(this->_internal_funcid());
  }

  // uint32 size = 4;
  if (this->_internal_size() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32SizePlusOne(this->_internal_size());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ScheduleReply::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    ScheduleReply::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ScheduleReply::GetClassData() const { return &_class_data_; }

void ScheduleReply::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<ScheduleReply *>(to)->MergeFrom(
      static_cast<const ScheduleReply &>(from));
}


void ScheduleReply::MergeFrom(const ScheduleReply& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:doinference.ScheduleReply)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_payload().empty()) {
    _internal_set_payload(from._internal_payload());
  }
  if (from._internal_guid() != 0) {
    _internal_set_guid(from._internal_guid());
  }
  if (from._internal_funcid() != 0) {
    _internal_set_funcid(from._internal_funcid());
  }
  if (from._internal_size() != 0) {
    _internal_set_size(from._internal_size());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ScheduleReply::CopyFrom(const ScheduleReply& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:doinference.ScheduleReply)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ScheduleReply::IsInitialized() const {
  return true;
}

void ScheduleReply::InternalSwap(ScheduleReply* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &payload_, lhs_arena,
      &other->payload_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(ScheduleReply, size_)
      + sizeof(ScheduleReply::size_)
      - PROTOBUF_FIELD_OFFSET(ScheduleReply, guid_)>(
          reinterpret_cast<char*>(&guid_),
          reinterpret_cast<char*>(&other->guid_));
}

::PROTOBUF_NAMESPACE_ID::Metadata ScheduleReply::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_infer_2eproto_getter, &descriptor_table_infer_2eproto_once,
      file_level_metadata_infer_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace doinference
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::doinference::ScheduleRequest* Arena::CreateMaybeMessage< ::doinference::ScheduleRequest >(Arena* arena) {
  return Arena::CreateMessageInternal< ::doinference::ScheduleRequest >(arena);
}
template<> PROTOBUF_NOINLINE ::doinference::ScheduleReply* Arena::CreateMaybeMessage< ::doinference::ScheduleReply >(Arena* arena) {
  return Arena::CreateMessageInternal< ::doinference::ScheduleReply >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
