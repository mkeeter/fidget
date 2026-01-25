pub(crate) fn new_buffer<T>(
    device: &wgpu::Device,
    name: impl AsRef<str>,
    count: usize,
    usages: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (std::mem::size_of::<T>() * count) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name.as_ref()),
        size,
        usage: usages,
        mapped_at_creation: false,
    })
}
