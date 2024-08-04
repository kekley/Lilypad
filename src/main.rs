use fps_counter::FPSCounter;
#[allow(unused)]
use glam::{
    f32::{Mat3, Vec3},
    Mat4,
};
use std::{borrow::Borrow, future::IntoFuture, sync::Arc, time::Duration};
use std::{
    collections::{HashMap, HashSet},
    iter::Map,
};
use vulkano::swapchain::SurfaceInfo;
use vulkano::sync;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferLevel,
        CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::GpuFuture,
    Validated, Version, VulkanError, VulkanLibrary,
};
use vulkano::{command_buffer::allocator::CommandBufferAllocator, format::Format};
use vulkano::{device::physical::PhysicalDevice, pipeline::graphics::viewport};
use vulkano::{instance::InstanceExtensions, swapchain::acquire_next_image};

use winit::{dpi::LogicalSize, event_loop, window::WindowId};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;
#[repr(C)]
#[derive(BufferContents, Vertex)]
pub struct MyVertex {
    #[format(R32G32B32A32_SFLOAT)]
    #[name("inPos")]
    pub pos: [f32; 4],
    #[name("inColor")]
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}
struct App {
    event_loop: EventLoop<()>,
    window: Arc<Window>,
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queues: HashMap<QueueFlags, Arc<Queue>>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    graphics_pipelines: HashMap<u32, Arc<GraphicsPipeline>>,
    viewport: Viewport,
    swapchain_image_views: Vec<Arc<ImageView>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    fps_counter: FPSCounter,
}

impl App {
    fn new(event_loop: EventLoop<()>) -> Self {
        let event_loop = event_loop;
        let window = Self::create_window(&event_loop);

        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Self::create_instance(required_extensions);

        let surface = Self::create_surface(&instance, &window);

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, index) =
            Self::select_physical_device(&instance, &surface, &device_extensions);

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, queue_map) =
            Self::create_device(physical_device.clone(), &device_extensions, index);

        let (swapchain, swapchain_images) =
            Self::create_swapchain(device.clone(), &surface, window.clone());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let mut graphics_pipelines: HashMap<u32, Arc<GraphicsPipeline>> = HashMap::default();
        graphics_pipelines.insert(
            0,
            Self::create_triangle_pipeline(device.clone(), vec![Some(swapchain.image_format())]),
        );
        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };
        let mut attachment_image_views =
            Self::window_size_dependent_setup(&swapchain_images, &mut viewport);
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let vertices = [
            MyVertex {
                pos: [-0.5, -0.25, 0.0, 1.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            MyVertex {
                pos: [0.0, 0.5, 0.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            MyVertex {
                pos: [0.25, -0.5, 0.0, 1.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ];

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();
        Self {
            event_loop: event_loop,
            window: window,
            instance: instance,
            surface: surface,
            physical_device: physical_device,
            device: device,
            queues: queue_map,
            swapchain: swapchain,
            swapchain_images: swapchain_images,
            memory_allocator: memory_allocator,
            graphics_pipelines: graphics_pipelines,
            viewport: viewport,
            swapchain_image_views: attachment_image_views,
            command_buffer_allocator: command_buffer_allocator,
            vertex_buffer: vertex_buffer,
            fps_counter: FPSCounter::new(),
        }
    }

    fn create_window(event_loop: &EventLoop<()>) -> Arc<Window> {
        let window_builder = WindowBuilder::new()
            .with_active(true)
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Vulkan");

        Arc::new(
            window_builder
                .build(&event_loop)
                .expect("Could not create window"),
        )
    }

    fn create_instance(required_extensions: InstanceExtensions) -> Arc<Instance> {
        let library = VulkanLibrary::new().expect("Could not find local vulkan library");

        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                application_name: Some("Peanits".to_string()),
                max_api_version: Some(Version::major_minor(1, 3)),
                ..Default::default()
            },
        )
        .expect("Could not create instance")
    }

    fn create_surface(instance: &Arc<Instance>, window: &Arc<Window>) -> Arc<Surface> {
        Surface::from_window(instance.clone(), window.clone()).expect("Could not create surface")
    }
    fn select_physical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface>,
        device_extensions: &DeviceExtensions,
    ) -> (Arc<PhysicalDevice>, u32) {
        instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| {
                p.supported_extensions().contains(device_extensions)
                    || p.api_version() >= Version::V1_3
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available")
    }

    fn get_exclusive_queue_index(
        physical_device: Arc<PhysicalDevice>,
        queue_flag: QueueFlags,
    ) -> Option<u32> {
        let index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, q)| {
                println!("Queue: {:?}, Flags: {:?}", i, q.queue_flags);
                if queue_flag == QueueFlags::TRANSFER {
                    q.queue_flags.contains(QueueFlags::TRANSFER)
                        && !q
                            .queue_flags
                            .intersects(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
                } else if queue_flag == QueueFlags::COMPUTE {
                    q.queue_flags.contains(QueueFlags::COMPUTE)
                        && !q.queue_flags.intersects(QueueFlags::GRAPHICS)
                } else {
                    q.queue_flags.contains(queue_flag)
                }
            });
        Some(index.expect("Could not find family") as u32)
    }

    fn create_device(
        physical_device: Arc<PhysicalDevice>,
        device_extensions: &DeviceExtensions,
        index: u32,
    ) -> (Arc<Device>, HashMap<QueueFlags, Arc<Queue>>) {
        let graphics_queue_index = index;
        let transfer_queue_index =
            Self::get_exclusive_queue_index(physical_device.clone(), QueueFlags::TRANSFER);

        let compute_queue_index =
            Self::get_exclusive_queue_index(physical_device.clone(), QueueFlags::COMPUTE);

        let mut queue_create_info_vec: Vec<QueueCreateInfo> = vec![];

        queue_create_info_vec.push(QueueCreateInfo {
            queue_family_index: index,
            ..Default::default()
        });

        match transfer_queue_index {
            Some(i) => {
                queue_create_info_vec.push(QueueCreateInfo {
                    queue_family_index: i,
                    ..Default::default()
                });
                println!("Transfer queue index: {}", i);
            }
            None => {}
        }
        match compute_queue_index {
            Some(i) => {
                queue_create_info_vec.push({
                    QueueCreateInfo {
                        queue_family_index: i,
                        ..Default::default()
                    }
                });
                println!("Compute queue index: {}", i);
            }
            None => {}
        }
        let create_info = DeviceCreateInfo {
            queue_create_infos: queue_create_info_vec,
            enabled_extensions: *device_extensions,
            enabled_features: Features {
                dynamic_rendering: true,
                ..Features::empty()
            },
            ..Default::default()
        };

        let (device, queues) = Device::new(physical_device.clone(), create_info)
            .expect("Could not create logical device");

        let mut queue_map: HashMap<QueueFlags, Arc<Queue>> = HashMap::new();
        queues.for_each(|q| {
            if q.queue_family_index() == graphics_queue_index {
                queue_map.insert(QueueFlags::GRAPHICS, q);
            } else {
                match transfer_queue_index {
                    Some(i) => {
                        if q.queue_family_index() == i {
                            queue_map.insert(QueueFlags::TRANSFER, q.clone());
                        }
                    }
                    None => {}
                }
                match compute_queue_index {
                    Some(i) => {
                        if q.queue_family_index() == i {
                            queue_map.insert(QueueFlags::COMPUTE, q.clone());
                        }
                    }
                    None => {}
                }
                {}
            }
        });
        (device, queue_map)
    }

    fn create_swapchain(
        device: Arc<Device>,
        surface: &Arc<Surface>,
        window: Arc<Window>,
    ) -> (
        Arc<vulkano::swapchain::Swapchain>,
        Vec<Arc<vulkano::image::Image>>,
    ) {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(surface, SurfaceInfo::default())
            .unwrap();

        let image_formats = device
            .physical_device()
            .surface_formats(surface, SurfaceInfo::default())
            .unwrap();

        let image_format = image_formats
            .iter()
            .enumerate()
            .find(|(_i, f)| f.0 == Format::R8G8B8A8_SRGB)
            .expect("Could not find 8 bit rgba color format")
            .1
             .0;

        Swapchain::new(
            device,
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .expect("Could not create swapchain")
    }

    fn create_triangle_pipeline(
        device: Arc<Device>,
        image_formats: Vec<Option<Format>>,
    ) -> Arc<GraphicsPipeline> {
        let vs = triangle_vertex_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = trinagle_fragment_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = MyVertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: image_formats,
            ..Default::default()
        };

        // Finally, create the pipeline.
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                // How vertex data is read from the vertex buffers into the vertex shader.
                vertex_input_state: Some(vertex_input_state),
                // How vertices are arranged into primitive shapes.
                // The default primitive shape is a triangle.
                input_assembly_state: Some(InputAssemblyState::default()),
                // How primitives are transformed and clipped to fit the framebuffer.
                // We use a resizable viewport, set to draw over the entire window.
                viewport_state: Some(ViewportState::default()),
                // How polygons are culled and converted into a raster of pixels.
                // The default value does not perform any culling.
                rasterization_state: Some(RasterizationState::default()),
                // How multiple fragment shader samples are converted to a single pixel value.
                // The default value does not perform any multisampling.
                multisample_state: Some(MultisampleState::default()),
                // How pixel values are combined with the values already present in the framebuffer.
                // The default value overwrites the old value with the new one, without any
                // blending.
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.color_attachment_formats.len() as u32,
                    ColorBlendAttachmentState::default(),
                )),
                // Dynamic states allows us to specify parts of the pipeline settings when
                // recording the command buffer, before we perform drawing.
                // Here, we specify that the viewport should be dynamic.
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }
    /// This function is called once during initialization, then again whenever the window is resized.
    fn window_size_dependent_setup(
        images: &[Arc<Image>],
        viewport: &mut Viewport,
    ) -> Vec<Arc<ImageView>> {
        let extent = images[0].extent();
        viewport.extent = [extent[0] as f32, extent[1] as f32];

        images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>()
    }

    fn main_loop(mut self) {
        let mut recreate_swapchain = false;

        let mut previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.event_loop.run(move |event, elwt, control_flow| {
            control_flow.set_poll();
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    control_flow.set_exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    recreate_swapchain = true;
                }
                Event::MainEventsCleared {} => {
                    let fps = self.fps_counter.tick();
                    self.window.set_title(format!("Vulkan FPS: {fps}").as_str());
                    let image_extent: [u32; 2] = self.window.inner_size().into();
                    if image_extent.contains(&0) {
                        return;
                    }

                    previous_frame_end.as_mut().unwrap().cleanup_finished(); //needs to be called to free gpu resources

                    if recreate_swapchain {
                        let (new_swapchain, images) = self
                            .swapchain
                            .recreate(SwapchainCreateInfo {
                                image_extent,
                                ..self.swapchain.create_info()
                            })
                            .expect("could not recreate swapchain");

                        let views = Self::window_size_dependent_setup(&images, &mut self.viewport);

                        self.swapchain = new_swapchain;
                        self.swapchain_images = images;
                        self.swapchain_image_views = views;
                        recreate_swapchain = false;
                    }

                    let (swap_image_index, suboptimal, acquire_future) = match acquire_next_image(
                        self.swapchain.clone(),
                        Some(Duration::from_secs(1)),
                    )
                    .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    // In order to draw, we have to record a *command buffer*. The command buffer object
                    // holds the list of commands that are going to be executed.
                    //
                    // Recording a command buffer is an expensive operation (usually a few hundred
                    // microseconds), but it is known to be a hot path in the driver and is expected to
                    // be optimized.
                    //
                    // Note that we have to pass a queue family when we create the command buffer. The
                    // command buffer will only be executable on that given queue family.
                    let mut builder = AutoCommandBufferBuilder::primary(
                        &self.command_buffer_allocator.clone(),
                        self.queues
                            .get(&QueueFlags::GRAPHICS)
                            .expect("no graphics queue")
                            .queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();

                    builder
                        // Before we can draw, we have to *enter a render pass*. We specify which
                        // attachments we are going to use for rendering here, which needs to match
                        // what was previously specified when creating the pipeline.
                        .begin_rendering(RenderingInfo {
                            // As before, we specify one color attachment, but now we specify the image
                            // view to use as well as how it should be used.
                            color_attachments: vec![Some(RenderingAttachmentInfo {
                                // `Clear` means that we ask the GPU to clear the content of this
                                // attachment at the start of rendering.
                                load_op: AttachmentLoadOp::Clear,
                                // `Store` means that we ask the GPU to store the rendered output in
                                // the attachment image. We could also ask it to discard the result.
                                store_op: AttachmentStoreOp::Store,
                                // The value to clear the attachment with. Here we clear it with a blue
                                // color.
                                //
                                // Only attachments that have `AttachmentLoadOp::Clear` are provided
                                // with clear values, any others should use `None` as the clear value.
                                clear_value: Some([0.0, 0.35, 0.500, 1.0].into()),
                                ..RenderingAttachmentInfo::image_view(
                                    // We specify image view corresponding to the currently acquired
                                    // swapchain image, to use for this attachment.
                                    self.swapchain_image_views[swap_image_index as usize].clone(),
                                )
                            })],
                            ..Default::default()
                        })
                        .unwrap()
                        // We are now inside the first subpass of the render pass.
                        //
                        // TODO: Document state setting and how it affects subsequent draw commands.
                        .set_viewport(0, [self.viewport.clone()].into_iter().collect())
                        .unwrap()
                        .bind_pipeline_graphics(
                            self.graphics_pipelines
                                .get(&0)
                                .expect("No graphics pipeline")
                                .clone(),
                        )
                        .unwrap()
                        .bind_vertex_buffers(0, self.vertex_buffer.clone())
                        .unwrap();

                    builder
                        // We add a draw command.
                        .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                        .unwrap();

                    builder
                        // We leave the render pass.
                        .end_rendering()
                        .unwrap();

                    // Finish recording the command buffer by calling `end`.
                    let command_buffer = builder.build().unwrap();

                    let future = previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(
                            self.queues.get(&QueueFlags::GRAPHICS).unwrap().clone(),
                            command_buffer,
                        )
                        .unwrap()
                        // The color output is now expected to contain our triangle. But in order to
                        // show it on the screen, we have to *present* the image by calling
                        // `then_swapchain_present`.
                        //
                        // This function does not actually present the image immediately. Instead it
                        // submits a present command at the end of the queue. This means that it will
                        // only be presented once the GPU has finished executing the command buffer
                        // that draws the triangle.
                        .then_swapchain_present(
                            self.queues.get(&QueueFlags::GRAPHICS).unwrap().clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                self.swapchain.clone(),
                                swap_image_index,
                            ),
                        )
                        .then_signal_fence_and_flush();
                    match future.map_err(Validated::unwrap) {
                        Ok(future) => {
                            previous_frame_end = Some(future.boxed());
                        }
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("failed to flush future: {e}");
                            previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                        }
                    }
                }
                _ => {
                    return;
                }
            }
        });
    }
}

mod triangle_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path:"shaders/triangle.vert",
    }
}

mod trinagle_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path:"shaders/triangle.frag",
    }
}
fn main() {
    let event_loop = EventLoop::new();
    let app = App::new(event_loop);
    app.main_loop();
}
