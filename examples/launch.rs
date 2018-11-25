#[macro_use]
extern crate rustacuda;

use rustacuda::prelude::*;
use std::ffi::CString;

fn main() {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
        .unwrap();

    let filename = CString::new("./resources/add.ptx").unwrap();
    let module = Module::load(&filename).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    // Create buffers for data
    let mut in_x = DeviceBuffer::from_slice(&[1.0f32; 10]).unwrap();
    let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10]).unwrap();
    let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10]).unwrap();
    let mut out_2 = DeviceBuffer::from_slice(&[0.0f32; 10]).unwrap();

    // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.sum<<<1, 1, 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_1.as_device_ptr(),
            out_1.len()
        ));
        result.unwrap();

        // Launch the kernel again using the `function` form:
        let function_name = CString::new("sum").unwrap();
        let sum = module.get_function(&function_name).unwrap();
        // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
        // configure grid and block size.
        let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_2.as_device_ptr(),
            out_2.len()
        ));
        result.unwrap();
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize().unwrap();

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 20];
    out_1.copy_to(&mut out_host[0..10]).unwrap();
    out_2.copy_to(&mut out_host[10..20]).unwrap();

    for x in out_host.iter() {
        assert_eq!(3.0 as u32, *x as u32);
    }

    println!("Launched kernel successfully.");
}
