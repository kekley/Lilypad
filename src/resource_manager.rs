use std::sync::Arc;
use std::u64;

use ash::vk::{
    AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
    Buffer, BufferCopy, BufferCreateInfo, BufferImageCopy, BufferUsageFlags, ColorComponentFlags,
    CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
    CommandBufferSubmitInfo, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags,
    CommandPoolCreateInfo, CompareOp, CullModeFlags, DependencyFlags, Extent2D, Extent3D, Fence,
    FenceCreateFlags, FenceCreateInfo, Filter, Format, Framebuffer, FramebufferCreateInfo,
    FrontFace, GraphicsPipelineCreateInfo, Image, ImageAspectFlags, ImageBlit, ImageCreateFlags,
    ImageCreateInfo, ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers,
    ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView, ImageViewCreateInfo,
    ImageViewType, LogicOp, MemoryPropertyFlags, Offset3D, Pipeline, PipelineCache,
    PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
    PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateFlags,
    PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineLayoutCreateFlags,
    PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateFlags, PipelineRasterizationStateCreateInfo,
    PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo, PipelineStageFlags,
    PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
    PrimitiveTopology, Queue, Rect2D, SampleCountFlags, ShaderModule, ShaderModuleCreateFlags,
    ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SubmitInfo, VertexInputRate, Viewport,
    REMAINING_ARRAY_LAYERS, REMAINING_MIP_LEVELS,
};
use ash::Device;
use vk_mem::{
    Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, Allocator, MemoryUsage,
};

use crate::rects::Extent;
use crate::vulkan_context::QueueFamiliesIndices;
pub struct ResourceManager {
    device: Device,
    transfer_queue: Queue,
    transfer_command_pool: CommandPool,
    transfer_command_buffer: CommandBuffer,
    immediate_command_fence: Fence,
    memory_allocator: Arc<Allocator>,
}

impl ResourceManager {
    pub fn new(
        device: &Device,
        queue_indices: &QueueFamiliesIndices,
        memory_allocator: Arc<Allocator>,
    ) -> Self {
        let transfer_command_pool = Self::create_command_pool(
            device,
            match queue_indices.transfer_index {
                Some(index) => index,
                None => queue_indices.graphics_index,
            },
            CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );

        let transfer_buffer_alloc_info = CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(transfer_command_pool)
            .level(CommandBufferLevel::PRIMARY);
        let transfer_command_buffer = unsafe {
            device
                .allocate_command_buffers(&transfer_buffer_alloc_info)
                .unwrap()[0]
        };

        let immediate_command_fence = unsafe {
            device
                .create_fence(
                    &FenceCreateInfo::default().flags(FenceCreateFlags::SIGNALED),
                    None,
                )
                .unwrap()
        };
        let transfer_queue = unsafe {
            match queue_indices.transfer_index {
                Some(i) => device.get_device_queue(i, 0),
                None => device.get_device_queue(queue_indices.graphics_index, 0),
            }
        };
        Self {
            memory_allocator: memory_allocator.clone(),
            transfer_queue: transfer_queue,
            device: device.clone(),
            transfer_command_pool: transfer_command_pool,
            transfer_command_buffer: transfer_command_buffer,
            immediate_command_fence: immediate_command_fence,
        }
    }
}
pub struct AllocatedBuffer {
    allocator: Arc<Allocator>,
    pub buffer: Buffer,
    allocation: Allocation,
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.buffer, &mut self.allocation)
        };
    }
}

impl AllocatedBuffer {
    fn new(
        size: u32,
        usage: BufferUsageFlags,
        memory_usage: MemoryUsage,
        allocator: Arc<Allocator>,
    ) -> Self {
        let buffer_create_info = BufferCreateInfo::default().size(size as u64).usage(usage);

        let alloc_create_info = AllocationCreateInfo {
            flags: AllocationCreateFlags::MAPPED
                | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,

            usage: memory_usage,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            allocator
                .create_buffer(&buffer_create_info, &alloc_create_info)
                .unwrap()
        };

        Self {
            allocator: allocator,
            buffer: buffer,
            allocation: allocation,
        }
    }
}

impl ResourceManager {
    fn create_command_pool(
        device: &Device,
        queue_family_index: u32,
        flags: CommandPoolCreateFlags,
    ) -> CommandPool {
        let create_info = CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(flags);

        let pool = unsafe { device.create_command_pool(&create_info, None).unwrap() };
        pool
    }
    pub fn immediate_transfer_submit(&self, func: impl Fn(CommandBuffer)) {
        unsafe {
            self.device
                .reset_fences(&[self.immediate_command_fence])
                .unwrap()
        };

        unsafe {
            self.device
                .begin_command_buffer(
                    self.transfer_command_buffer,
                    &CommandBufferBeginInfo::default()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap()
        };
        func(self.transfer_command_buffer);
        unsafe {
            self.device
                .end_command_buffer(self.transfer_command_buffer)
                .unwrap()
        };

        let binding = [self.transfer_command_buffer];
        let submit = SubmitInfo::default().command_buffers(&binding);

        unsafe {
            self.device
                .queue_submit(self.transfer_queue, &[submit], self.immediate_command_fence)
                .unwrap()
        };
        unsafe {
            self.device
                .wait_for_fences(&[self.immediate_command_fence], true, u64::MAX)
        };
    }
}

pub struct AllocatedImage {
    device: ash::Device,
    pub allocator: Arc<vk_mem::Allocator>,
    pub extent: Extent,
    pub image: ash::vk::Image,
    pub allocation: vk_mem::Allocation,
    pub view: ash::vk::ImageView,
}

impl AllocatedImage {
    fn new(
        device: &ash::Device,
        allocator: Arc<Allocator>,
        extent: Extent,
        mip_levels: u32,
        sample_count: SampleCountFlags,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        aspect_mask: ImageAspectFlags,
    ) -> Self {
        let vk_extent = Extent3D {
            width: extent.width(),
            height: extent.height(),
            depth: 1,
        };
        let image_info = ImageCreateInfo::default()
            .image_type(ImageType::TYPE_2D)
            .extent(vk_extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .samples(sample_count)
            .flags(ImageCreateFlags::empty());
        let mut allocation_info: AllocationCreateInfo = AllocationCreateInfo::default();
        allocation_info.usage = MemoryUsage::AutoPreferDevice;
        allocation_info.required_flags = MemoryPropertyFlags::DEVICE_LOCAL;
        let (image, allocation) = unsafe {
            allocator
                .create_image(&image_info, &allocation_info)
                .unwrap()
        };
        let view = Self::create_image_view(device, image, format, aspect_mask);
        AllocatedImage {
            extent,
            device: device.clone(),
            allocator: allocator,
            image,
            allocation,
            view,
        }
    }
    pub fn blit_image(
        device: &ash::Device,
        cmd: CommandBuffer,
        src: Image,
        dst: Image,
        src_size: [i32; 2],
        dst_size: [i32; 2],
    ) {
        let range = ImageSubresourceLayers::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0);
        let src_offsets = [
            Offset3D::default(),
            Offset3D::default().x(src_size[0]).y(src_size[1]).z(1),
        ];
        let dst_offsets = [
            Offset3D::default(),
            Offset3D::default().x(dst_size[0]).y(dst_size[1]).z(1),
        ];
        let blit = ImageBlit::default()
            .src_offsets(src_offsets)
            .src_subresource(range)
            .dst_offsets(dst_offsets)
            .dst_subresource(range);
        let filter = Filter::LINEAR;
        unsafe {
            device.cmd_blit_image(
                cmd,
                src,
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                filter,
            )
        };
    }
    pub fn blit_image_preserve_ratio(
        device: &ash::Device,
        cmd: CommandBuffer,
        src: Image,
        dst: Image,
        src_size: [i32; 2],
        dst_size: [i32; 2],
    ) {
        let range = ImageSubresourceLayers::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0);
        let src_offsets = [
            Offset3D::default(),
            Offset3D::default().x(src_size[0]).y(src_size[1]).z(1),
        ];

        let dst_offsets = [
            Offset3D::default(),
            Offset3D::default().x(dst_size[0]).y(dst_size[1]).z(1),
        ];
        let blit = ImageBlit::default()
            .src_offsets(src_offsets)
            .src_subresource(range)
            .dst_offsets(dst_offsets)
            .dst_subresource(range);
        let filter = Filter::LINEAR;
        unsafe {
            device.cmd_blit_image(
                cmd,
                src,
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                filter,
            )
        };
    }

    pub fn create_image_view(
        device: &ash::Device,
        image: Image,
        format: Format,
        aspect_mask: ImageAspectFlags,
    ) -> ImageView {
        let create_info = ImageViewCreateInfo::default()
            .image(image)
            .view_type(ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe { device.create_image_view(&create_info, None).unwrap() }
    }

    pub fn transition_image(
        device: &ash::Device,
        cmd: CommandBuffer,
        image: Image,
        initial_layout: ImageLayout,
        final_layout: ImageLayout,
    ) {
        let aspect_mask = {
            if final_layout == ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL
            } else {
                ImageAspectFlags::COLOR
            }
        };
        let image_barrier = ImageMemoryBarrier::default()
            .src_access_mask(AccessFlags::MEMORY_WRITE)
            .dst_access_mask(AccessFlags::MEMORY_WRITE | AccessFlags::MEMORY_READ)
            .old_layout(initial_layout)
            .new_layout(final_layout)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_array_layer(0)
                    .base_mip_level(0)
                    .level_count(REMAINING_MIP_LEVELS)
                    .layer_count(REMAINING_ARRAY_LAYERS),
            )
            .image(image);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                PipelineStageFlags::ALL_COMMANDS,
                PipelineStageFlags::ALL_COMMANDS,
                DependencyFlags::default(),
                &[],
                &[],
                &[image_barrier],
            )
        };
    }
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);

            self.allocator
                .destroy_image(self.image, &mut self.allocation);
        };
    }
}

impl ResourceManager {
    pub fn create_buffer(
        &self,
        size: u32,
        usage: BufferUsageFlags,
        memory_usage: MemoryUsage,
    ) -> AllocatedBuffer {
        AllocatedBuffer::new(size, usage, memory_usage, self.memory_allocator.clone())
    }
    pub fn create_image_from_data(
        &self,
        extent: Extent,
        mip_levels: u32,
        sample_count: SampleCountFlags,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        aspect_mask: ImageAspectFlags,
        data: *const u8,
        data_size: usize,
    ) -> AllocatedImage {
        let image = self.create_image(
            extent,
            mip_levels,
            sample_count,
            format,
            tiling,
            usage | ImageUsageFlags::TRANSFER_DST,
            aspect_mask,
        );
        let staging_buffer = self.create_and_upload_buffer(
            data_size as u32,
            BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferDevice,
            data,
        );

        self.copy_buffer_to_image(&staging_buffer, &image);
        image
    }
    pub fn create_image(
        &self,
        extent: Extent,
        mip_levels: u32,
        sample_count: SampleCountFlags,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        aspect_mask: ImageAspectFlags,
    ) -> AllocatedImage {
        AllocatedImage::new(
            &self.device,
            self.memory_allocator.clone(),
            extent,
            mip_levels,
            sample_count,
            format,
            tiling,
            usage,
            aspect_mask,
        )
    }
    pub fn copy_buffer_to_image(&self, buffer: &AllocatedBuffer, image: &AllocatedImage) {
        let copy_func = |cmd: CommandBuffer| {
            let copy = BufferImageCopy::default()
                .image_subresource(
                    ImageSubresourceLayers::default()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(image.extent.to_vk_extent_3d());
            AllocatedImage::transition_image(
                &self.device,
                cmd,
                image.image,
                ImageLayout::UNDEFINED,
                ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    cmd,
                    buffer.buffer,
                    image.image,
                    ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                )
            }
            AllocatedImage::transition_image(
                &self.device,
                cmd,
                image.image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        };
        self.immediate_transfer_submit(copy_func);
    }
    pub fn create_and_upload_buffer(
        &self,
        size: u32,
        usage: BufferUsageFlags,
        memory_usage: MemoryUsage,
        data: *const u8,
    ) -> AllocatedBuffer {
        let mut staging_buffer = AllocatedBuffer::new(
            size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            self.memory_allocator.clone(),
        );
        let new_buffer = AllocatedBuffer::new(
            size,
            usage | BufferUsageFlags::TRANSFER_DST,
            memory_usage,
            self.memory_allocator.clone(),
        );

        let mapped = unsafe {
            self.memory_allocator
                .map_memory(&mut staging_buffer.allocation)
                .unwrap()
        };

        unsafe { mapped.copy_from_nonoverlapping(data, size as usize) };

        let copy_func = |cmd: CommandBuffer| {
            let copy = BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size as u64);
            unsafe {
                self.device
                    .cmd_copy_buffer(cmd, staging_buffer.buffer, new_buffer.buffer, &[copy])
            };
        };

        self.immediate_transfer_submit(copy_func);

        unsafe {
            self.memory_allocator
                .unmap_memory(&mut staging_buffer.allocation)
        };
        new_buffer
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_command_pool(self.transfer_command_pool, None);

            self.device
                .destroy_fence(self.immediate_command_fence, None);
        };
    }
}
