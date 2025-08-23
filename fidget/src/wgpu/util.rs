use zerocopy::{Immutable, IntoBytes};

pub(crate) fn write_storage_buffer<T: IntoBytes + Immutable>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &mut wgpu::Buffer,
    name: &str,
    data: &[T],
) {
    let size = std::mem::size_of_val(data) as u64;
    if size != buf.size() {
        *buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    queue
        .write_buffer_with(
            buf,
            0,
            size.try_into().expect("buffer size must be > 0"),
        )
        .unwrap()
        .copy_from_slice(data.as_bytes())
}

pub(crate) fn resize_buffer_with<T>(
    device: &wgpu::Device,
    buf: &mut wgpu::Buffer,
    name: &str,
    count: usize,
    usages: wgpu::BufferUsages,
) -> bool {
    let size = (std::mem::size_of::<T>() * count) as u64;
    if size != buf.size() {
        *buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: usages,
            mapped_at_creation: false,
        });
        true
    } else {
        false
    }
}
