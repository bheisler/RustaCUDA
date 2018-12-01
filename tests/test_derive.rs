#![allow(dead_code)]

#[macro_use]
extern crate rustacuda;
extern crate rustacuda_core;

#[derive(Clone, DeviceCopy)]
struct ZeroSizedStruct;

#[derive(Clone, DeviceCopy)]
struct TupleStruct(u64, u64);

#[derive(Clone, DeviceCopy)]
struct NormalStruct {
    x: u64,
    y: u64,
}

#[derive(Clone, DeviceCopy)]
struct ContainerStruct {
    a: NormalStruct,
    b: TupleStruct,
}

#[derive(Clone, DeviceCopy)]
struct GenericStruct<T> {
    value: T,
}

#[derive(Clone, DeviceCopy)]
enum TestEnum {
    Unit,
    Tuple(u64),
    Struct { x: u64, y: u64 },
    Container { a: NormalStruct, b: TupleStruct },
}

#[derive(Clone, DeviceCopy)]
enum GenericEnum<T> {
    Unit,
    Generic { val: T },
}

#[derive(Copy, Clone, DeviceCopy)]
#[repr(C)]
union TestUnion {
    u: u64,
    i: i64,
}

#[test]
fn test_hidden_functions() {
    __verify_ZeroSizedStruct_can_implement_DeviceCopy(&ZeroSizedStruct);
    __verify_TupleStruct_can_implement_DeviceCopy(&TupleStruct(0, 0));
    __verify_NormalStruct_can_implement_DeviceCopy(&NormalStruct { x: 0, y: 0 });
    __verify_ContainerStruct_can_implement_DeviceCopy(&ContainerStruct {
        a: NormalStruct { x: 0, y: 0 },
        b: TupleStruct(0, 0),
    });
    __verify_GenericStruct_can_implement_DeviceCopy(&GenericStruct { value: 0u64 });
    __verify_TestEnum_can_implement_DeviceCopy(&TestEnum::Unit);
    __verify_GenericEnum_can_implement_DeviceCopy::<u64>(&GenericEnum::Unit);
    __verify_TestUnion_can_implement_DeviceCopy(&TestUnion { u: 0u64 });
}
