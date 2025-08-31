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
