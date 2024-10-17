use crate::pipeline_builder::GraphicsPipelineBuilder;
use crate::rects::Extent;
use crate::render_frame::RenderFrame;
use crate::resource_manager::{AllocatedImage, ResourceManager};
use crate::shaders::{Shader, TRIANGLE_FRAGMENT_SHADER_CODE, TRIANGLE_VERTEX_SHADER_CODE};
use crate::swapchain::{SwapchainProperties, SwapchainSupportDetails};
use crate::vulkan_context::{QueueFamiliesIndices, VulkanContext};
use crate::{App, FRAMES_IN_FLIGHT, HEIGHT, MSAA_SAMPLES, WIDTH};
use ash::util::read_spv;
use ash::vk::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Buffer,
    BufferCopy, BufferCreateInfo, BufferImageCopy, BufferUsageFlags, ColorComponentFlags,
    CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
    CommandBufferSubmitInfo, CommandBufferUsageFlags, CompareOp, CullModeFlags, Fence,
    FenceCreateFlags, FenceCreateInfo, Framebuffer, FramebufferCreateInfo, FrontFace,
    GraphicsPipelineCreateInfo, ImageSubresourceLayers, LogicOp, Pipeline, PipelineCache,
    PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
    PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateFlags,
    PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineLayoutCreateFlags,
    PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateFlags, PipelineRasterizationStateCreateInfo,
    PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo,
    PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
    PrimitiveTopology, Rect2D, ShaderModule, ShaderModuleCreateFlags, ShaderModuleCreateInfo,
    ShaderStageFlags, SubmitInfo, VertexInputRate, Viewport,
};
use ash::Instance;
use ash::{
    vk::{
        self, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, CompositeAlphaFlagsKHR,
        Extent2D, Extent3D, Format, Image, ImageAspectFlags, ImageCreateFlags, ImageCreateInfo,
        ImageLayout, ImageTiling, ImageType, ImageUsageFlags, ImageView, MemoryPropertyFlags,
        Queue, RenderPass, SampleCountFlags, SharingMode, SwapchainCreateInfoKHR, SwapchainKHR,
    },
    Device,
};
use load_file;
use load_file::load_bytes;
use std::ffi::{CStr, CString};
use std::io::Cursor;
use std::sync::Arc;
use vk_mem::{
    self, Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, Allocator,
    AllocatorCreateInfo, MemoryUsage,
};

pub struct AppResources {
    swapchain: ash::khr::swapchain::Device,
    swapchain_khr: SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    swapchain_images: Vec<Image>,
    swapchain_image_views: Vec<ImageView>,
    render_pass: RenderPass,
    default_pipeline_layout: PipelineLayout,
    default_pipeline: Pipeline,
    depth_image: AllocatedImage,
    draw_image: AllocatedImage,
    resolve_image: AllocatedImage,
    framebuffer: Framebuffer,
    frames: Vec<RenderFrame>,
    context: VulkanContext,
    device: Device,
}

impl AppResources {
    pub fn swapchain_properties(&self) -> SwapchainProperties {
        self.swapchain_properties
    }
    pub fn render_pass(&self) -> RenderPass {
        self.render_pass
    }
    pub fn framebuffer(&self) -> Framebuffer {
        self.framebuffer
    }
    pub fn frame(&mut self, frame_number: usize) -> &mut RenderFrame {
        assert!(frame_number < FRAMES_IN_FLIGHT as usize);
        &mut self.frames[frame_number]
    }
    pub fn swapchain(&self) -> ash::khr::swapchain::Device {
        self.swapchain.clone()
    }
    pub fn swapchain_khr(&self) -> SwapchainKHR {
        self.swapchain_khr
    }
    pub fn draw_image(&self) -> &AllocatedImage {
        &self.draw_image
    }
    pub fn depth_image(&self) -> &AllocatedImage {
        &self.depth_image
    }
    pub fn resolve_image(&self) -> Image {
        self.resolve_image.image
    }
    pub fn swapchain_images(&self) -> Vec<Image> {
        self.swapchain_images.clone()
    }
    pub fn pipeline(&self) -> Pipeline {
        self.default_pipeline
    }
}

impl AppResources {
    pub fn device(&self) -> Device {
        self.device.clone()
    }
    pub fn context(&self) -> &VulkanContext {
        &self.context
    }
}

impl AppResources {
    pub fn new(context: VulkanContext, resource_manager: Arc<ResourceManager>) -> Self {
        let (swapchain, swapchain_khr, images, swapchain_properties) =
            Self::create_swapchain_and_images(
                context.physical_device(),
                context.device(),
                context.instance(),
                context.surface(),
                context.surface_khr(),
                context.queue_indices(),
                &[WIDTH, HEIGHT],
            );
        let image_views =
            Self::create_swapchain_image_views(context.device(), &images, swapchain_properties);
        let render_pass = Self::create_render_pass(
            context.device(),
            swapchain_properties,
            MSAA_SAMPLES,
            Format::D24_UNORM_S8_UINT,
        );

        let draw_image = Self::create_draw_image(resource_manager.clone());

        let depth_image = Self::create_depth_image(resource_manager.clone());

        let resolve_image = Self::create_resolve_image(resource_manager.clone());

        let frames = {
            let mut vec = vec![];
            for i in 0..FRAMES_IN_FLIGHT {
                vec.push(RenderFrame::new(&context));
            }
            vec
        };

        let framebuffer = unsafe {
            let binding = [draw_image.view, depth_image.view, resolve_image.view];

            let framebuffer_create_info = FramebufferCreateInfo::default()
                .attachment_count(3)
                .render_pass(render_pass)
                .attachments(&binding)
                .width(WIDTH)
                .height(HEIGHT)
                .layers(1);
            context
                .device()
                .create_framebuffer(&framebuffer_create_info, None)
                .unwrap()
        };

        let (pipeline_layout, pipeline) = Self::create_pipeline(&context.device(), render_pass);
        Self {
            device: context.device().clone(),
            swapchain: swapchain,
            swapchain_khr: swapchain_khr,
            swapchain_properties,
            swapchain_images: images,
            swapchain_image_views: image_views,
            render_pass: render_pass,
            default_pipeline_layout: pipeline_layout,
            default_pipeline: pipeline,
            draw_image: draw_image,
            depth_image: depth_image,
            resolve_image: resolve_image,
            framebuffer: framebuffer,
            frames: frames,
            context: context,
        }
    }

    pub fn recreate_swapchain(&mut self, dimensions: &[u32; 2]) {
        unsafe {
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);

            for view in self.swapchain_image_views.clone() {
                self.context.device().destroy_image_view(view, None);
            }
            let (swapchain, swapchain_khr, images, properties) = Self::create_swapchain_and_images(
                self.context.physical_device(),
                self.context.device(),
                self.context.instance(),
                self.context.surface(),
                self.context.surface_khr(),
                self.context.queue_indices(),
                dimensions,
            );
            let views =
                Self::create_swapchain_image_views(self.context.device(), &images, properties);
            self.swapchain_image_views = views;
            self.swapchain_images = images;
            self.swapchain_properties = properties;
            self.swapchain = swapchain;
            self.swapchain_khr = swapchain_khr;
        };
    }
    fn create_pipeline(device: &Device, render_pass: RenderPass) -> (PipelineLayout, Pipeline) {
        let vert_shader = Shader::new(
            device,
            ShaderStageFlags::VERTEX,
            TRIANGLE_VERTEX_SHADER_CODE,
        );

        let frag_shader = Shader::new(
            device,
            ShaderStageFlags::FRAGMENT,
            TRIANGLE_FRAGMENT_SHADER_CODE,
        );

        let mut builder = GraphicsPipelineBuilder::new(vert_shader, frag_shader, None, None);

        builder.enable_msaa();
        let (pipeline_layout, pipeline) = builder.build(device, render_pass);
        (pipeline_layout, pipeline)
    }
    fn create_depth_image(resource_manager: Arc<ResourceManager>) -> AllocatedImage {
        let format = Format::D24_UNORM_S8_UINT;
        let render_extent = Extent::new(WIDTH, HEIGHT);
        let depth_image = resource_manager.create_image(
            render_extent,
            1,
            MSAA_SAMPLES,
            format,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ImageAspectFlags::DEPTH,
        );
        depth_image
    }
    fn create_draw_image(resource_manager: Arc<ResourceManager>) -> AllocatedImage {
        let render_extent = Extent::new(WIDTH, HEIGHT);

        let draw_image = resource_manager.create_image(
            render_extent,
            1,
            MSAA_SAMPLES,
            Format::R16G16B16A16_SFLOAT,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::COLOR_ATTACHMENT,
            ImageAspectFlags::COLOR,
        );
        draw_image
    }

    fn create_resolve_image(resource_manager: Arc<ResourceManager>) -> AllocatedImage {
        let render_extent = Extent::new(WIDTH, HEIGHT);

        resource_manager.create_image(
            render_extent,
            1,
            SampleCountFlags::TYPE_1,
            Format::R16G16B16A16_SFLOAT,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
            ImageAspectFlags::COLOR,
        )
    }

    fn create_swapchain_image_views(
        device: &ash::Device,
        swapchain_images: &Vec<Image>,
        properties: SwapchainProperties,
    ) -> Vec<ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                AllocatedImage::create_image_view(
                    &device,
                    *image,
                    properties.format.format,
                    ImageAspectFlags::COLOR,
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment_desc = vk::AttachmentDescription::default()
            .format(Format::R16G16B16A16_SFLOAT)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_attachement_desc = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
            .stencil_store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let attachment_resolve = AttachmentDescription::default()
            .format(Format::R16G16B16A16_SFLOAT)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::DONT_CARE)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::TRANSFER_SRC_OPTIMAL);

        let attachment_descs = [
            color_attachment_desc,
            depth_attachement_desc,
            attachment_resolve,
        ];

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_refs = [color_attachment_ref];

        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_attachment_ref = AttachmentReference::default()
            .attachment(2)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_attachment_refs = [resolve_attachment_ref];

        let subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .resolve_attachments(&resolve_attachment_refs);
        let subpass_descs = [subpass_desc];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs);

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }
    fn create_swapchain_and_images(
        physical_device: &ash::vk::PhysicalDevice,
        device: &Device,
        instance: &Instance,
        surface: &ash::khr::surface::Instance,
        surface_khr: &ash::vk::SurfaceKHR,
        queue_family_indices: &QueueFamiliesIndices,
        dimensions: &[u32; 2],
    ) -> (
        ash::khr::swapchain::Device,
        SwapchainKHR,
        Vec<Image>,
        SwapchainProperties,
    ) {
        let details = SwapchainSupportDetails::new(*physical_device, &surface, *surface_khr);
        let properties = details.get_ideal_swapchain_properties(*dimensions);

        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;
        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        println!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count,
        );

        let graphics = queue_family_indices.graphics_index;
        let present = queue_family_indices.present_index;
        let indices = [graphics, present];

        let create_info = {
            let mut builder = SwapchainCreateInfoKHR::default()
                .surface(*surface_khr)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_DST);
            builder = if graphics != present {
                builder
                    .image_sharing_mode(SharingMode::CONCURRENT)
                    .queue_family_indices(&indices)
            } else {
                builder.image_sharing_mode(SharingMode::EXCLUSIVE)
            };
            builder
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };
        let swapchain = ash::khr::swapchain::Device::new(&instance, &device);

        let swapchain_khr = unsafe {
            swapchain
                .create_swapchain(&create_info, None)
                .expect("Could not create swapchain")
        };

        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        (swapchain, swapchain_khr, images, properties)
    }
}

impl Drop for AppResources {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_render_pass(self.render_pass, None);
            self.context
                .device()
                .destroy_framebuffer(self.framebuffer, None);

            self.context
                .device()
                .destroy_pipeline_layout(self.default_pipeline_layout, None);
            self.context
                .device()
                .destroy_pipeline(self.default_pipeline, None);

            self.swapchain.destroy_swapchain(self.swapchain_khr, None);

            for view in self.swapchain_image_views.clone() {
                self.context.device().destroy_image_view(view, None);
            }
        };
    }
}
