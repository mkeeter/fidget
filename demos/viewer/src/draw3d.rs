use crate::{CustomTexture, Mode3D};
use eframe::{
    egui,
    egui_wgpu::{self, wgpu},
};
use fidget::render::GeometryPixel;
use zerocopy::{Immutable, IntoBytes};

/// Configuration for 3D rendering with geometry data
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, IntoBytes, Immutable)]
pub(crate) struct RenderConfig {
    render_mode: u32, // 0 = heightmap, 1 = shaded
    max_depth: u32,
}

pub(crate) struct Draw3D {
    geometry_pipeline: wgpu::RenderPipeline,
    tex: Option<(fidget::render::ImageSize, Vec<CustomTexture>)>,
    geometry_sampler: wgpu::Sampler,
    geometry_bind_group_layout: wgpu::BindGroupLayout,

    /// Local copy of render configuration
    render_config: RenderConfig,

    /// GPU buffer for render configuration
    render_config_buffer: wgpu::Buffer,
}

impl Draw3D {
    pub fn init(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        // Create Geometry shader module
        let geometry_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Geometry Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/geometry.wgsl").into(),
                ),
            });

        let geometry_sampler =
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Geometry Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest, // Use nearest for integer textures
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

        // Create bind group layout for Geometry texture and sampler
        let geometry_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Geometry Bind Group Layout"),
                entries: &[
                    // Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::NonFiltering,
                        ),
                        count: None,
                    },
                    // Uniform buffer for render configuration
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let geometry_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Geometry Render Pipeline Layout"),
                bind_group_layouts: &[&geometry_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create the Geometry render pipeline
        let geometry_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Geometry Render Pipeline"),
                layout: Some(&geometry_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &geometry_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &geometry_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::OVER,
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        // Create a buffer for render configuration
        let render_config = RenderConfig::default();
        let render_config_buffer =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Render Config Buffer"),
                size: std::mem::size_of::<RenderConfig>() as u64,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        Draw3D {
            geometry_pipeline,
            tex: None,
            geometry_sampler,
            geometry_bind_group_layout,
            render_config,
            render_config_buffer,
        }
    }
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: &[Vec<GeometryPixel>],
        image_size: fidget::render::ImageSize,
        mode: Mode3D,
        max_depth: u32,
    ) {
        let (width, height) = (image_size.width(), image_size.height());
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Function to create a new Geometry texture
        let new_geometry_tex = || -> CustomTexture {
            // Create the texture - we need to store depth and normals,
            // so we'll use a format that can store 4 32-bit values
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Geometry Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Create the texture view
            let texture_view =
                texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create the bind group for this texture
            let bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Geometry Bind Group"),
                    layout: &self.geometry_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &self.geometry_sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(
                                wgpu::BufferBinding {
                                    buffer: &self.render_config_buffer,
                                    offset: 0,
                                    size: None,
                                },
                            ),
                        },
                    ],
                });

            CustomTexture {
                bind_group,
                texture,
            }
        };

        // Check to see whether we can reuse textures
        match &mut self.tex {
            Some((tex_size, tex_data)) if *tex_size == image_size => {
                tex_data.resize_with(images.len(), new_geometry_tex);
            }
            Some(..) | None => {
                let textures =
                    images.iter().map(|_i| new_geometry_tex()).collect();
                self.tex = Some((image_size, textures));
            }
        }

        // Update render config locally, then copy to the GPU
        self.render_config.render_mode = match mode {
            Mode3D::Heightmap => 0,
            Mode3D::Color => 1,
            Mode3D::Shaded => 2,
        };
        self.render_config.max_depth = max_depth;
        queue.write_buffer(
            &self.render_config_buffer,
            0,
            self.render_config.as_bytes(),
        );

        // Upload all of the images to textures
        for (image_data, tex) in
            images.iter().zip(self.tex.as_ref().unwrap().1.iter())
        {
            // Upload geometry data using AsBytes trait
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                image_data.as_bytes(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16 * width), // 16 bytes per GeometryPixel (4 u32 values)
                    rows_per_image: Some(height),
                },
                texture_size,
            );
        }
    }

    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Only draw if we have a texture
        if let Some((_tex_size, tex_data)) = &self.tex {
            for tex in tex_data {
                render_pass.set_pipeline(&self.geometry_pipeline);
                render_pass.set_bind_group(0, &tex.bind_group, &[]);

                // Draw 2 triangles (6 vertices) to form a quad
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}

pub(crate) struct Draw3DCallback {
    data: Option<(Vec<Vec<GeometryPixel>>, fidget::render::ImageSize)>,
    mode: Mode3D,
    max_depth: u32,
}

impl Draw3DCallback {
    pub fn new(
        data: Option<(Vec<Vec<GeometryPixel>>, fidget::render::ImageSize)>,
        mode: Mode3D,
        max_depth: u32,
    ) -> Self {
        Self {
            data,
            mode,
            max_depth,
        }
    }
}

impl egui_wgpu::CallbackTrait for Draw3DCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &mut Draw3D = resources.get_mut().unwrap();
        if let Some((image_data, image_size)) = self.data.as_ref() {
            resources.prepare(
                device,
                queue,
                image_data.as_slice(),
                *image_size,
                self.mode,
                self.max_depth,
            )
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &Draw3D = resources.get().unwrap();
        resources.paint(render_pass);
    }
}
