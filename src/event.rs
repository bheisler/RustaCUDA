//! Events to track status and dependencies, and measure duration of work
//! submitted to a CUDA stream.
//!
//! In CUDA, most work is performed asynchronously. Events help to manage tasks
//! scheduled on an asynchronous stream. This includes waiting for a task (or
//! multiple tasks) to complete, and measuring the time duration it takes to
//! complete a task. Events can also be used to sequence tasks on multiple
//! streams within the same context by specifying dependent tasks (not supported
//! yet by RustaCUDA).
//!
//! Events may be reused multiple times.

use crate::cuda_sys::cuda::{
    cuEventCreate, cuEventDestroy_v2, cuEventElapsedTime, cuEventQuery, cuEventRecord,
    cuEventSynchronize, CUevent,
};
use crate::error::{CudaError, CudaResult, DropResult, ToResult};
use crate::stream::Stream;

use std::mem;
use std::ptr;

bitflags! {
    /// Bit flags for configuring a CUDA Event.
    ///
    /// Nvidia claims that setting `DISABLE_TIMING` and `BLOCKING_SYNC` provides
    /// the best performance for `query()` and `stream.wait_event()`.
    pub struct EventFlags: u32 {
        /// The default event creation flag.
        const DEFAULT = 0x0;

        /// Specify that the created event should busy-wait on blocking
        /// function calls.
        const BLOCKING_SYNC = 0x1;

        /// Specify that the created event does not need to record timing data.
        const DISABLE_TIMING = 0x2;

        /// Specify that the created event may be used as an interprocess event.
        /// (not supported yet by RustaCUDA). This flag requires
        /// `DISABLE_TIMING` to be set as well.
        const INTERPROCESS = 0x4;
    }
}

/// Status enum that represents the current status of an event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventStatus {
    /// Ready indicates that all work captured by the event has been completed.
    ///
    /// NVidia states that for Unified Memory, `EventStatus::Ready` is
    /// equivalent to having called `Event::synchronize`.
    Ready,

    /// `EventStatus::NotReady` indicates that the work captured by the event is still
    /// incomplete.
    NotReady,
}

/// An event to track work submitted to a stream.
///
/// See the module-level documentation for more information.
#[derive(Debug)]
pub struct Event(CUevent);

impl Event {
    /// Create a new event with the specified flags.
    ///
    /// # Example:
    ///
    /// ```
    /// # use rustacuda::quick_init;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _context = quick_init()?;
    /// use rustacuda::event::{Event, EventFlags};
    ///
    /// // With default settings
    /// let event = Event::new(EventFlags::DEFAULT)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(flags: EventFlags) -> CudaResult<Self> {
        unsafe {
            let mut event: CUevent = mem::zeroed();
            cuEventCreate(&mut event, flags.bits()).to_result()?;
            Ok(Event(event))
        }
    }

    /// Record the state of a stream at the time of the call.
    ///
    /// This function is used together with `query`, `synchronize`, and
    /// `elapsed_time_f32`. See the respective function for more information.
    ///
    /// If the event is created with `EventFlags::BLOCKING_SYNC`, then `record`
    /// blocks until the event has actually been recorded.
    ///
    /// # Errors
    ///
    /// If the event and stream are not from the same context, an error is
    /// returned.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustacuda::quick_init;
    /// # use rustacuda::stream::{Stream, StreamFlags};
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _context = quick_init()?;
    /// use rustacuda::event::{Event, EventFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let event = Event::new(EventFlags::DEFAULT)?;
    ///
    /// // do some work ...
    ///
    /// event.record(&stream)?;
    /// # Ok(())
    /// }
    /// ```
    pub fn record(&self, stream: &Stream) -> CudaResult<()> {
        unsafe {
            cuEventRecord(self.0, stream.as_inner()).to_result()?;
            Ok(())
        }
    }

    /// Return the current status of the event.
    ///
    /// The status captures the contents of the stream underlying the event. If
    /// the stream is empty, `query` will return `EventStatus::Ready`.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustacuda::quick_init;
    /// # use rustacuda::stream::{Stream, StreamFlags};
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _context = quick_init()?;
    /// use rustacuda::event::{Event, EventFlags, EventStatus};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let event = Event::new(EventFlags::DEFAULT)?;
    ///
    /// // do some work ...
    ///
    /// // record an event
    /// event.record(&stream)?;
    ///
    /// // ... wait some time ...
    /// # event.synchronize()?;
    ///
    /// // query if the work is finished
    /// let status = event.query()?;
    /// assert_eq!(status, EventStatus::Ready);
    /// # Ok(())
    /// }
    /// ```
    pub fn query(&self) -> CudaResult<EventStatus> {
        let result = unsafe { cuEventQuery(self.0).to_result() };

        match result {
            Ok(()) => Ok(EventStatus::Ready),
            Err(CudaError::NotReady) => Ok(EventStatus::NotReady),
            Err(other) => Err(other),
        }
    }

    /// Wait for an event to complete.
    ///
    /// Blocks thread execution until all work submitted before the event was
    /// recorded has completed. `EventFlags::BLOCKING_SYNC` controls the mode of
    /// blocking. If the flag is set on event creation, the thread will sleep.
    /// Otherwise, the thread will busy-wait.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustacuda::quick_init;
    /// # use rustacuda::stream::{Stream, StreamFlags};
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _context = quick_init()?;
    /// use rustacuda::event::{Event, EventFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let event = Event::new(EventFlags::DEFAULT)?;
    ///
    /// // do some work ...
    ///
    /// // record an event
    /// event.record(&stream)?;
    ///
    /// // wait until the work is finished
    /// event.synchronize()?;
    /// # Ok(())
    /// }
    /// ```
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe {
            cuEventSynchronize(self.0).to_result()?;
            Ok(())
        }
    }

    /// Return the duration between two events.
    ///
    /// The duration is computed in milliseconds with a resulution of
    /// approximatly 0.5 microseconds. As `record` takes place asynchronously,
    /// other work can be scheduled in between the events, thus yieling an
    /// unexpectedly long duration.
    ///
    /// # Errors
    ///
    /// `CudaError::NotReady` is returned if either event is not yet complete.
    ///
    /// `CudaError::InvalidHandle` is returned if
    /// - the two events are not from the same context, or if
    /// - `record` has not been called on either event, or if
    /// - the `DISABLE_TIMING` flag is set on either event.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustacuda::quick_init;
    /// # use rustacuda::stream::{Stream, StreamFlags};
    /// # use rustacuda::launch;
    /// # use rustacuda::module::Module;
    /// # use rustacuda::memory::DeviceBox;
    /// # use std::error::Error;
    /// # use std::ffi::CString;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _context = quick_init()?;
    /// # let module_data = CString::new(include_str!("../resources/add.ptx"))?;
    /// # let module = Module::load_from_string(&module_data)?;
    /// # let mut x = DeviceBox::new(&10.0f32)?;
    /// # let mut y = DeviceBox::new(&20.0f32)?;
    /// # let mut result = DeviceBox::new(&0.0f32)?;
    /// use rustacuda::event::{Event, EventFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let start_event = Event::new(EventFlags::DEFAULT)?;
    /// let stop_event = Event::new(EventFlags::DEFAULT)?;
    ///
    /// // start recording time
    /// start_event.record(&stream)?;
    ///
    /// // do some work ...
    /// # unsafe {
    /// #    launch!(module.sum<<<1, 1, 0, stream>>>(
    /// #            x.as_device_ptr(),
    /// #            y.as_device_ptr(),
    /// #            result.as_device_ptr(),
    /// #            1 // Length
    /// #            ))?;
    /// # }
    ///
    /// // stop recording time
    /// stop_event.record(&stream)?;
    ///
    /// // wait for the work to complete
    /// stop_event.synchronize()?;
    ///
    /// // compute the time elapsed between the start and stop events
    /// let time = stop_event.elapsed_time_f32(&start_event)?;
    ///
    /// # assert!(time > 0.0);
    /// # Ok(())
    /// }
    /// ```
    pub fn elapsed_time_f32(&self, start: &Self) -> CudaResult<f32> {
        unsafe {
            let mut millis: f32 = 0.0;
            cuEventElapsedTime(&mut millis, start.0, self.0).to_result()?;
            Ok(millis)
        }
    }

    // Get the inner `CUevent` from the `Event`.
    //
    // Necessary for certain CUDA functions outside of this
    // module that expect a bare `CUevent`.
    pub(crate) fn as_inner(&self) -> CUevent {
        self.0
    }

    /// Destroy an `Event` returning an error.
    ///
    /// Destroying an event can return errors from previous asynchronous work.
    /// This function destroys the given event and returns the error and the
    /// un-destroyed event on failure.
    ///
    /// # Example:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _context = quick_init()?;
    /// use rustacuda::event::{Event, EventFlags};
    ///
    /// let event = Event::new(EventFlags::DEFAULT)?;
    /// match Event::drop(event) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((cuda_error, event)) => {
    ///         println!("Failed to destroy event: {:?}", cuda_error);
    ///         // Do something with event
    ///     },
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn drop(mut event: Event) -> DropResult<Event> {
        if event.0.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut event.0, ptr::null_mut());
            match cuEventDestroy_v2(inner).to_result() {
                Ok(()) => {
                    mem::forget(event);
                    Ok(())
                }
                Err(e) => Err((e, Event(inner))),
            }
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cuEventDestroy_v2(self.0) }
            .to_result()
            .expect("Failed to destroy CUDA event");
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::quick_init;
    use crate::stream::StreamFlags;
    use std::error::Error;

    #[test]
    fn test_new_with_flags() -> Result<(), Box<dyn Error>> {
        let _context = quick_init()?;
        let _event = Event::new(EventFlags::BLOCKING_SYNC | EventFlags::DISABLE_TIMING)?;
        Ok(())
    }

    #[test]
    fn test_record_with_wrong_context() -> Result<(), Box<dyn Error>> {
        let _context = quick_init()?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let _new_context = quick_init()?;
        let event = Event::new(EventFlags::DEFAULT)?;
        let result = event.record(&stream);
        assert_eq!(result, Err(CudaError::InvalidHandle));
        Ok(())
    }

    #[test]
    fn test_elapsed_time_f32_with_wrong_context() -> Result<(), Box<dyn Error>> {
        let _context = quick_init()?;
        let fst_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let fst_event = Event::new(EventFlags::DEFAULT)?;
        fst_event.record(&fst_stream)?;

        let _context = quick_init()?;
        let snd_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let snd_event = Event::new(EventFlags::DEFAULT)?;
        snd_event.record(&snd_stream)?;

        fst_event.synchronize()?;
        snd_event.synchronize()?;
        let result = snd_event.elapsed_time_f32(&fst_event);
        assert_eq!(result, Err(CudaError::InvalidHandle));
        Ok(())
    }

    #[test]
    fn test_elapsed_time_f32_with_different_streams() -> Result<(), Box<dyn Error>> {
        let _context = quick_init()?;
        let fst_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let fst_event = Event::new(EventFlags::DEFAULT)?;
        fst_event.record(&fst_stream)?;

        let snd_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let snd_event = Event::new(EventFlags::DEFAULT)?;
        snd_event.record(&snd_stream)?;

        fst_event.synchronize()?;
        snd_event.synchronize()?;
        let _result = snd_event.elapsed_time_f32(&fst_event)?;
        Ok(())
    }

    #[test]
    fn test_elapsed_time_f32_with_disable_timing() -> Result<(), Box<dyn Error>> {
        let _context = quick_init()?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let start_event = Event::new(EventFlags::DISABLE_TIMING)?;
        start_event.record(&stream)?;

        let stop_event = Event::new(EventFlags::DEFAULT)?;
        stop_event.record(&stream)?;

        stop_event.synchronize()?;
        let result = stop_event.elapsed_time_f32(&start_event);
        assert_eq!(result, Err(CudaError::InvalidHandle));
        Ok(())
    }
}