use std::{collections::VecDeque, rc::Rc};

use ash::vk::{
    self, Buffer, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool,
    DescriptorPoolCreateInfo, DescriptorPoolResetFlags, DescriptorPoolSize, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorType, ImageLayout, ImageView,
    Sampler, WriteDescriptorSet,
};

pub struct DescriptorWriter<'a> {
    image_writes: Vec<DescriptorImageWrite<'a>>,
    buffer_writes: Vec<DescriptorBufferWrite<'a>>,
}

pub struct DescriptorBufferWrite<'a> {
    buffer_infos: Vec<DescriptorBufferInfo>,
    write: WriteDescriptorSet<'a>,
}

pub struct DescriptorImageWrite<'a> {
    image_infos: Vec<DescriptorImageInfo>,
    write: WriteDescriptorSet<'a>,
}

impl DescriptorWriter<'_> {
    pub fn new() -> Self {
        Self {
            image_writes: Vec::default(),
            buffer_writes: Vec::default(),
        }
    }

    pub fn write_image(
        &mut self,
        binding: u32,
        image_view: ImageView,
        sampler: Sampler,
        image_layout: ImageLayout,
        descriptor_type: DescriptorType,
    ) {
        let mut image_write = DescriptorImageWrite {
            image_infos: vec![DescriptorImageInfo {
                sampler,
                image_view,
                image_layout,
            }],
            write: WriteDescriptorSet::default()
                .descriptor_type(descriptor_type)
                .dst_binding(binding)
                .descriptor_count(1),
        };
        image_write.write.p_image_info = image_write.image_infos.as_ptr();
        self.image_writes.push(image_write)
    }

    pub fn write_texture_array(
        &mut self,
        binding: u32,
        image_views: Vec<ImageView>,
        image_layout: ImageLayout,
        descriptor_type: DescriptorType,
    ) {
        let mut image_infos = Vec::new();
        for view in &image_views {
            let image_info = DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(image_layout)
                .sampler(Sampler::null());
            image_infos.push(image_info);
        }
        let mut texture_array_write = DescriptorImageWrite {
            image_infos: image_infos,
            write: WriteDescriptorSet::default()
                .descriptor_type(descriptor_type)
                .dst_binding(binding)
                .descriptor_count(image_views.len() as u32),
        };
        texture_array_write.write.p_image_info = texture_array_write.image_infos.as_ptr();
        self.image_writes.push(texture_array_write);
    }
    
    pub fn write_buffer(
        &mut self,
        binding: u32,
        buffer: Buffer,
        size: u64,
        offset: u64,
        descriptor_type: DescriptorType,
    ) {
        let mut buffer_write = DescriptorBufferWrite {
            buffer_infos: vec![DescriptorBufferInfo {
                buffer,
                offset,
                range: size,
            }],
            write: WriteDescriptorSet::default()
                .dst_binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(1),
        };
        buffer_write.write.p_buffer_info = buffer_write.buffer_infos.as_ptr();
        self.buffer_writes.push(buffer_write);
    }

    pub fn update(&mut self, device: &ash::Device, set: DescriptorSet) {
        let mut writes = Vec::with_capacity(self.buffer_writes.len() + self.image_writes.len());
        for buffer_write in &mut self.buffer_writes {
            buffer_write.write.dst_set = set;
            writes.push(buffer_write.write);
        }
        for image_write in &mut self.image_writes {
            image_write.write.dst_set = set;
            writes.push(image_write.write)
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
    pub fn clear(&mut self) {
        self.buffer_writes.clear();
        self.image_writes.clear();
    }
}

#[derive(Clone, Copy)]
pub struct PoolSizeRatio {
    pub descriptor_type: ash::vk::DescriptorType,
    pub ratio: f32,
}

pub struct DescriptorAllocator {
    device: ash::Device,
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<DescriptorPool>,
    ready_pools: Vec<DescriptorPool>,
    sets_per_pool: u32,
}
impl DescriptorAllocator {
    pub fn new(device: &ash::Device, initial_sets: u32, pool_ratios: &[PoolSizeRatio]) -> Self {
        assert!(pool_ratios.len() > 0);
        let mut ratios: Vec<PoolSizeRatio> = vec![];
        for ratio in pool_ratios {
            ratios.push(*ratio);
        }
        let init_pool = Self::create_pool(device.clone(), initial_sets, pool_ratios);
        Self {
            device: device.clone(),
            ratios: ratios,
            full_pools: Vec::default(),
            ready_pools: vec![init_pool],
            sets_per_pool: (initial_sets as f32 * 1.5) as u32,
        }
    }

    fn destroy_pools(&mut self) {
        for pool in &self.ready_pools {
            unsafe { self.device.destroy_descriptor_pool(*pool, None) };
        }
        self.ready_pools.clear();
        for pool in &self.full_pools {
            unsafe { self.device.destroy_descriptor_pool(*pool, None) };
        }
        self.full_pools.clear()
    }
    pub fn clear_pools(&mut self) {
        for pool in &self.ready_pools {
            unsafe {
                self.device
                    .reset_descriptor_pool(*pool, DescriptorPoolResetFlags::default())
                    .unwrap()
            };
        }

        for pool in &self.full_pools {
            unsafe {
                self.device
                    .reset_descriptor_pool(*pool, DescriptorPoolResetFlags::default())
                    .unwrap();
                self.ready_pools.push(*pool);
            };
        }
        self.full_pools.clear();
    }

    pub fn allocate(&mut self, layout: DescriptorSetLayout) -> DescriptorSet {
        let mut pool_to_use = self.get_pool();

        let layouts = [layout];
        let alloc_info = DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool_to_use)
            .set_layouts(&layouts);

        let descriptor_set;
        match unsafe { self.device.allocate_descriptor_sets(&alloc_info) } {
            Ok(set) => descriptor_set = set,
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.full_pools.push(pool_to_use);
                pool_to_use = self.get_pool();
                let alloc_info = alloc_info.descriptor_pool(pool_to_use);
                unsafe {
                    descriptor_set = self
                        .device
                        .allocate_descriptor_sets(&alloc_info)
                        .expect("could not allocate descriptor set after new pool creation");
                };
            }
            Err(err) => {
                panic!("could not allocate descriptor set, Cause: {}", err);
            }
        }
        self.ready_pools.push(pool_to_use);
        descriptor_set[0]
    }

    fn get_pool(&mut self) -> DescriptorPool {
        let pool;
        match self.ready_pools.pop() {
            Some(ready_pool) => {
                pool = ready_pool;
            }
            None => {
                pool = Self::create_pool(self.device.clone(), self.sets_per_pool, &self.ratios);
                self.sets_per_pool = (self.sets_per_pool as f32 * 1.5) as u32;
                if self.sets_per_pool > 4092 {
                    self.sets_per_pool = 4092
                }
            }
        }
        pool
    }

    pub fn create_pool(
        device: ash::Device,
        set_count: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> DescriptorPool {
        let mut pool_sizes: Vec<ash::vk::DescriptorPoolSize> =
            Vec::with_capacity(pool_ratios.len());
        for ratio in pool_ratios {
            pool_sizes.push(DescriptorPoolSize {
                ty: ratio.descriptor_type,
                descriptor_count: (ratio.ratio * set_count as f32) as u32,
            })
        }
        let create_info = DescriptorPoolCreateInfo::default()
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        self.destroy_pools();
    }
}
