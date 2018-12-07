extern crate rustacuda;

use rustacuda::prelude::*;
use rustacuda::quick_init;
use std::sync::mpsc::sync_channel;

#[test]
fn test_stream_callbacks_execution_order() {
    let _ctx = quick_init();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let (order_sender, order_receiver) = sync_channel(0);
    stream
        .add_callback(Box::new(|_| {
            order_sender.send(1).unwrap();
        }))
        .unwrap();
    stream
        .add_callback(Box::new(|_| {
            order_sender.send(2).unwrap();
        }))
        .unwrap();
    stream
        .add_callback(Box::new(|_| {
            order_sender.send(3).unwrap();
        }))
        .unwrap();
    for expected in &[1, 2, 3] {
        assert_eq!(*expected, order_receiver.recv().unwrap());
    }
}

#[test]
fn test_stream_callbacks_environment_capture() {
    let _ctx = quick_init();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let (capture_sender, capture_receiver) = sync_channel(0);
    let magic_numbers = (42, Box::new(1337));
    stream
        .add_callback(Box::new(|_| {
            capture_sender.send(magic_numbers).unwrap();
        }))
        .unwrap();
    let captured_magic_numbers = capture_receiver.recv().unwrap();
    assert_eq!(42, captured_magic_numbers.0);
    assert_eq!(1337, *captured_magic_numbers.1);
}

#[test]
fn test_stream_callbacks_status_propagation() {
    let _ctx = quick_init();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let (status_sender, status_receiver) = sync_channel(0);
    stream
        .add_callback(Box::new(|status| {
            status_sender.send(status).unwrap();
        }))
        .unwrap();
    assert_eq!(Ok(()), status_receiver.recv().unwrap())
}
