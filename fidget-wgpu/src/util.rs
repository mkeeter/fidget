pub(crate) fn new_buffer<T>(
    device: &wgpu::Device,
    name: &str,
    count: usize,
    usages: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (std::mem::size_of::<T>() * count) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name),
        size,
        usage: usages,
        mapped_at_creation: false,
    })
}
