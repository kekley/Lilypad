use std::cmp::min;

use ash::vk::{
    self, CommandBuffer, CommandBufferAllocateInfo, CommandBufferLevel, CommandPool,
    CommandPoolCreateFlags, CommandPoolCreateInfo, DescriptorPool, DescriptorPoolCreateInfo,
    DescriptorPoolResetFlags, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo,
    DescriptorSetLayout, DescriptorType, Fence, FenceCreateFlags, FenceCreateInfo, Image,
    Semaphore, SemaphoreCreateInfo,
};

use crate::{
    descriptors::{DescriptorAllocator, PoolSizeRatio},
    resource_manager::{AllocatedBuffer, AllocatedImage},
    vulkan_context::VulkanContext,
};

pub struct RenderFrame {
    device: ash::Device,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    descriptor_allocator: DescriptorAllocator,
    swapchain_semaphore: Semaphore,
    render_semaphore: Semaphore,
    fence: Fence,
    current_frame_buffers: Vec<AllocatedBuffer>,
    current_frame_images: Vec<AllocatedImage>,
}

impl RenderFrame {
    pub fn push_image(&mut self, image: AllocatedImage) {
        self.current_frame_images.push(image);
    }
    pub fn push_buffer(&mut self, buffer: AllocatedBuffer) {
        self.current_frame_buffers.push(buffer);
    }
    pub fn clear_frame_data(&mut self) {
        self.current_frame_buffers.clear();
        self.current_frame_images.clear();
    }
    pub fn command_buffer(&self) -> CommandBuffer {
        self.command_buffer
    }
    pub fn fence(&self) -> Fence {
        self.fence
    }
    pub fn swapchain_semaphore(&self) -> Semaphore {
        self.swapchain_semaphore
    }
    pub fn render_semaphore(&self) -> Semaphore {
        self.render_semaphore
    }
    pub fn descriptor_set(&mut self, layout: DescriptorSetLayout) -> DescriptorSet {
        self.descriptor_allocator.allocate(layout)
    }
}

impl RenderFrame {
    pub fn new(context: &VulkanContext) -> Self {
        let create_info = CommandPoolCreateInfo::default()
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(context.queue_indices().graphics_index);
        let pool = unsafe {
            context
                .device()
                .create_command_pool(&create_info, None)
                .unwrap()
        };

        let allocate_info = CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(1)
            .level(CommandBufferLevel::PRIMARY);

        let buffer = unsafe {
            context
                .device()
                .allocate_command_buffers(&allocate_info)
                .unwrap()[0]
                .to_owned()
        };

        let fence_info = FenceCreateInfo::default().flags(FenceCreateFlags::SIGNALED);
        let semaphore_info = SemaphoreCreateInfo::default();

        let fence = unsafe { context.device().create_fence(&fence_info, None).unwrap() };
        let swapchain_semaphore = unsafe {
            context
                .device()
                .create_semaphore(&semaphore_info, None)
                .unwrap()
        };
        let render_semaphore = unsafe {
            context
                .device()
                .create_semaphore(&semaphore_info, None)
                .unwrap()
        };

        let pool_ratios = [
            PoolSizeRatio {
                descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
                ratio: 4.0,
            },
            PoolSizeRatio {
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                ratio: 3.0,
            },
        ];
        Self {
            current_frame_buffers: Vec::default(),
            current_frame_images: Vec::default(),
            device: context.device().clone(),
            command_pool: pool,
            command_buffer: buffer,
            descriptor_allocator: DescriptorAllocator::new(context.device(), 1000, &pool_ratios),
            swapchain_semaphore: swapchain_semaphore,
            render_semaphore: render_semaphore,
            fence: fence,
        }
    }
}

impl Drop for RenderFrame {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_semaphore(self.render_semaphore, None);
            self.device
                .destroy_semaphore(self.swapchain_semaphore, None);
            self.device.destroy_fence(self.fence, None);
        };
    }
}
