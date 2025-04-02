use crate::CustomTexture;
use eframe::{
    egui,
    egui_wgpu::{self, wgpu},
};
use zerocopy::IntoBytes;

struct Resources {
    rgba_pipeline: wgpu::RenderPipeline,
    tex: Option<(fidget::render::ImageSize, Vec<CustomTexture>)>,
    rgba_sampler: wgpu::Sampler,
    rgba_bind_group_layout: wgpu::BindGroupLayout,
}

impl Resources {
    fn init(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Create RGBA shader module
        let rgba_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RGBA Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/image.wgsl").into(),
                ),
            });

        // Create samplers
        let rgba_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("RGBA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create bind group layout for RGBA texture and sampler
        let rgba_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RGBA Bind Group Layout"),
                entries: &[
                    // Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
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
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        // Create render pipeline layouts
        let rgba_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RGBA Render Pipeline Layout"),
                bind_group_layouts: &[&rgba_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create the RGBA render pipeline
        let rgba_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("RGBA Render Pipeline"),
                layout: Some(&rgba_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &rgba_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &rgba_shader,
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
        Resources {
            rgba_pipeline,
            tex: None,
            rgba_sampler,
            rgba_bind_group_layout,
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: &[Vec<[u8; 4]>],
        image_size: fidget::render::ImageSize,
    ) {
        let (width, height) = (image_size.width(), image_size.height());
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Function to create a new RGBA texture
        let new_rgba_tex = || -> CustomTexture {
            // Create the texture
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("RGBA Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
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
                    label: Some("RGBA Bind Group"),
                    layout: &self.rgba_bind_group_layout,
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
                                &self.rgba_sampler,
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
                tex_data.resize_with(images.len(), new_rgba_tex);
            }
            Some(..) | None => {
                let textures = images.iter().map(|_i| new_rgba_tex()).collect();
                self.tex = Some((image_size, textures));
            }
        }

        // Upload all of the images to textures
        for (image_data, tex) in
            images.iter().zip(self.tex.as_ref().unwrap().1.iter())
        {
            // Upload RGBA image data
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
                    bytes_per_row: Some(4 * width),
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
                render_pass.set_pipeline(&self.rgba_pipeline);
                render_pass.set_bind_group(0, &tex.bind_group, &[]);

                // Draw 2 triangles (6 vertices) to form a quad
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}

/// GPU callback to render 2D RGBA images
pub(crate) struct Draw2D {
    data: Option<(Vec<Vec<[u8; 4]>>, fidget::render::ImageSize)>,
}

impl Draw2D {
    pub fn new(
        data: Option<(Vec<Vec<[u8; 4]>>, fidget::render::ImageSize)>,
    ) -> Self {
        Self { data }
    }

    pub fn init(wgpu_state: &eframe::egui_wgpu::RenderState) {
        let resources =
            Resources::init(&wgpu_state.device, wgpu_state.target_format);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);
    }
}

impl egui_wgpu::CallbackTrait for Draw2D {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &mut Resources = resources.get_mut().unwrap();
        if let Some((image_data, image_size)) = self.data.as_ref() {
            resources.prepare(device, queue, image_data.as_slice(), *image_size)
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &Resources = resources.get().unwrap();
        resources.paint(render_pass);
    }
}
