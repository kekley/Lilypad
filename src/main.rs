mod app_resources;
mod debug;
mod descriptors;
mod font;
mod pipeline_builder;
mod rects;
mod render_frame;
mod renderer;
mod resource_manager;
mod shaders;
mod swapchain;
mod tile;
mod tile_map;
mod tile_set;
mod vulkan_context;
use std::{
    borrow::BorrowMut,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::Result;
use app_resources::AppResources;
use ash::vk::{
    self, ClearColorValue, ClearDepthStencilValue, ClearValue, CommandBufferBeginInfo,
    CommandBufferResetFlags, CommandBufferUsageFlags, Extent2D, Extent3D, Filter, Framebuffer,
    Image, ImageAspectFlags, ImageBlit, ImageCopy, ImageLayout, ImageSubresource,
    ImageSubresourceLayers, ImageSubresourceRange, Offset2D, Offset3D, PipelineBindPoint,
    PipelineStageFlags, PresentInfoKHR, Rect2D, RenderPassBeginInfo, RenderingAttachmentInfo,
    RenderingInfo, SampleCountFlags, SubmitInfo, SubpassContents, REMAINING_ARRAY_LAYERS,
    REMAINING_MIP_LEVELS,
};
use fps_counter::FPSCounter;

use rects::Extent;
use renderer::{RenderContext, Renderer2D};
use resource_manager::{AllocatedImage, ResourceManager};
use swapchain::{SwapchainProperties, SwapchainSupportDetails};
use tile_map::TileMap;
use vk_mem::{Allocator, AllocatorCreateInfo};
use vulkan_context::{QueueFamiliesIndices, VulkanContext};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};
pub const WIDTH: u32 = 1600;
pub const HEIGHT: u32 = 900;
pub const MSAA_SAMPLES: SampleCountFlags = SampleCountFlags::TYPE_4;
pub const FRAMES_IN_FLIGHT: usize = 2;
fn main() {
    pretty_env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    window: Option<Window>,
    window_resized: bool,
    renderer: Option<Renderer2D>,
    resource_manager: Option<Arc<ResourceManager>>,
    resources: Option<AppResources>,
    tile_map: TileMap,
    frame_count: usize,
    last_frame_time: u128,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let tile_map = TileMap::load_from_png("../assets/r.png".to_string());
        Self {
            tile_map: tile_map,
            window: None,
            resource_manager: None,
            window_resized: false,
            resources: None,
            renderer: None,
            frame_count: 0,
            last_frame_time: 0,
        }
    }
    fn initialize(&mut self) {
        let context = VulkanContext::new(self.window.as_ref().unwrap());
        let memory_allocator = unsafe {
            Arc::new(
                Allocator::new(AllocatorCreateInfo::new(
                    context.instance(),
                    context.device(),
                    *context.physical_device(),
                ))
                .unwrap(),
            )
        };
        self.resource_manager = Some(Arc::new(ResourceManager::new(
            context.device(),
            context.queue_indices(),
            memory_allocator,
        )));
        self.resources = Some(AppResources::new(
            context,
            self.resource_manager.clone().unwrap(),
        ));
        self.renderer = Some(Renderer2D::new(
            &self.resources.as_ref().unwrap().device(),
            self.resource_manager.clone().unwrap(),
            self.resources.as_ref().unwrap().render_pass(),
        ));
        self.load_scene();
    }
    fn load_scene(&mut self) {
        self.renderer.as_mut().unwrap().add_sprite_sheet(
            "../textures/tilemap.png".to_string(),
            Extent::new(18, 18),
            Some(Extent::new(1, 1)),
        );
    }
    fn render_frame(&mut self) {
        if self.window_resized {
            self.window_resized = false;
            self.resources
                .as_mut()
                .unwrap()
                .recreate_swapchain(&self.window.as_ref().unwrap().inner_size().into());
        }
        let current_frame = self.frame_count % FRAMES_IN_FLIGHT;
        let resources = self.resources.as_mut().unwrap();
        let device = resources.device();
        let swapchain = resources.swapchain();
        let swapchain_khr = resources.swapchain_khr();
        let swapchain_images = resources.swapchain_images();
        let swapchain_properties = resources.swapchain_properties();
        let render_pass = resources.render_pass();
        let draw_image = resources.draw_image();
        let depth_image = resources.depth_image();
        let resolve_image_image = resources.resolve_image();
        let pipeline = resources.pipeline();
        let framebuffer = resources.framebuffer();
        let queues = resources.context().queues();
        let frame_data = resources.frame(current_frame);
        let render_semaphore = frame_data.render_semaphore();
        let swapchain_semaphore = frame_data.swapchain_semaphore();
        let cmd = frame_data.command_buffer();

        unsafe {
            device
                .wait_for_fences(&[frame_data.fence()], true, 1000000000)
                .unwrap();
        };

        frame_data.clear_frame_data();

        unsafe {
            device.reset_fences(&[frame_data.fence()]).unwrap();
        };

        let (swap_image_index, resized) = unsafe {
            swapchain
                .acquire_next_image(
                    swapchain_khr,
                    1000000000,
                    frame_data.swapchain_semaphore(),
                    ash::vk::Fence::null(),
                )
                .unwrap()
        };
        if resized {
            self.window_resized = true;
            return;
        }

        let swap_image = swapchain_images[swap_image_index as usize];
        unsafe {
            device
                .reset_command_buffer(cmd, CommandBufferResetFlags::empty())
                .unwrap()
        };
        let begin_info =
            CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin_info).unwrap() };

        let clear_values = [
            ClearValue {
                color: ClearColorValue {
                    float32: [0.5, 0.5, 0.5, 1.0],
                },
            },
            ClearValue {
                depth_stencil: ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];
        let render_pass_begin = RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .clear_values(&clear_values)
            .render_area(
                Rect2D::default()
                    .offset(Offset2D::default())
                    .extent(Extent2D::default().width(WIDTH).height(HEIGHT)),
            );
        let cmds = [cmd];
        let signals = [render_semaphore];
        let waits = [swapchain_semaphore];
        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_begin, SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd, PipelineBindPoint::GRAPHICS, pipeline);
            //device.cmd_draw(cmd, 3, 1, 0, 0);

            self.renderer
                .as_mut()
                .unwrap()
                .draw_tile_map(&self.tile_map);

            device.cmd_end_render_pass(cmd);

            AllocatedImage::transition_image(
                &device,
                cmd,
                swap_image,
                ImageLayout::UNDEFINED,
                ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            AllocatedImage::blit_image(
                &device,
                cmd,
                resolve_image_image,
                swap_image,
                [WIDTH as i32, HEIGHT as i32],
                [
                    swapchain_properties.extent.width as i32,
                    swapchain_properties.extent.height as i32,
                ],
            );

            AllocatedImage::transition_image(
                &device,
                cmd,
                swap_image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                ImageLayout::PRESENT_SRC_KHR,
            );
            device.end_command_buffer(cmd).unwrap();

            let submits = SubmitInfo::default()
                .command_buffers(&cmds)
                .signal_semaphores(&signals)
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]);
            device
                .queue_submit(queues.graphics_queue, &[submits], frame_data.fence())
                .unwrap();
        }

        let indices = [swap_image_index];
        let swapchains = [resources.swapchain_khr()];
        let present_info = PresentInfoKHR::default()
            .wait_semaphores(&signals)
            .image_indices(&indices)
            .swapchains(&swapchains);

        let result = unsafe {
            resources
                .swapchain()
                .queue_present(resources.context().queues().present_queue, &present_info)
        };
        match result {
            Ok(false) => {}
            Ok(true) => self.window_resized = true,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.window_resized = true,
            Err(error) => {
                panic!("failed to present. Cause: {}", error)
            }
        }
        self.frame_count += 1;
    }
    fn main_loop(&mut self, last_frame_time: u128) -> Duration {
        let start = Instant::now();

        if self.frame_count % 60 == 0 {
            println!(
                "last frame time: {} , FPS: {}",
                last_frame_time,
                1.0 / (last_frame_time as f32 / 1000000 as f32) as f32
            );
        }
        self.update_scene();
        self.render_frame();
        let end = Instant::now();

        let total_time = end - start;
        total_time
    }
    fn update_scene(&self) {}
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
                        .with_title("Vulkan")
                        .with_resizable(false),
                )
                .unwrap(),
        );
        self.initialize();
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                unsafe {
                    self.resources
                        .as_ref()
                        .unwrap()
                        .device()
                        .device_wait_idle()
                        .unwrap()
                };
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let frame_time = self.main_loop(self.last_frame_time).as_micros();
                self.last_frame_time = frame_time as u128;
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(_size) => self.window_resized = true,
            _ => {}
        }
    }
}
