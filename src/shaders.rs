use std::fmt::Debug;

use ash::vk::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
    PipelineShaderStageCreateInfo, ShaderModuleCreateFlags, ShaderModuleCreateInfo,
    ShaderStageFlags,
};
use inline_spirv::include_spirv;

pub const TRIANGLE_VERTEX_SHADER_CODE: &[u32] =
    include_spirv!("./shaders/triangle.vert", vert, glsl, entry = "main");
pub const TRIANGLE_FRAGMENT_SHADER_CODE: &[u32] =
    include_spirv!("./shaders/triangle.frag", frag, glsl, entry = "main");
pub const SPRITE_VERTEX_SHADER_CODE: &[u32] =
    include_spirv!("./shaders/sprite.vert", vert, glsl, entry = "main");
pub const SPRITE_FRAGMENT_SHADER_CODE: &[u32] =
    include_spirv!("./shaders/sprite.frag", frag, glsl, entry = "main");
pub const FONT_VERTEX_SHADER_CODE: &[u32] =
    include_spirv!("./shaders/font.vert", vert, glsl, entry = "main");
pub const FONT_FRAGMENT_SHADER_CODE: &[u32] =
    include_spirv!("./shaders/font.frag", frag, glsl, entry = "main");

const ENTRY: &core::ffi::CStr = c"main";

#[derive(Debug, Default)]
pub struct Shader {
    device: Option<ShaderDevice>,
    shader_stage: ash::vk::ShaderStageFlags,
    module: ash::vk::ShaderModule,
}

struct ShaderDevice(ash::Device);

impl Debug for ShaderDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("handle", &self.0.handle())
            .finish()
    }
}

impl Shader {
    pub fn new(
        device: &ash::Device,
        shader_stage: ash::vk::ShaderStageFlags,
        code: &[u32],
    ) -> Self {
        let create_info = ShaderModuleCreateInfo::default()
            .code(code)
            .flags(ShaderModuleCreateFlags::empty());
        let module = unsafe { device.create_shader_module(&create_info, None).unwrap() };
        Self {
            device: Some(ShaderDevice(device.clone())),
            shader_stage: shader_stage,
            module: module,
        }
    }
    pub fn pipeline_stage_create_info(&self) -> ash::vk::PipelineShaderStageCreateInfo {
        PipelineShaderStageCreateInfo::default()
            .name(ENTRY)
            .module(self.module)
            .stage(self.shader_stage)
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .0
                .destroy_shader_module(self.module, None)
        };
    }
}

pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<DescriptorSetLayoutBinding<'a>>,
}

impl DescriptorLayoutBuilder<'_> {
    pub fn new() -> Self {
        Self {
            bindings: Vec::default(),
        }
    }
    pub fn add_binding(
        &mut self,
        binding: u32,
        stage_flags: ShaderStageFlags,
        descriptor_count: u32,
        descriptor_type: DescriptorType,
    ) {
        self.bindings.push(
            DescriptorSetLayoutBinding::default()
                .binding(binding)
                .stage_flags(stage_flags)
                .descriptor_count(descriptor_count)
                .descriptor_type(descriptor_type),
        )
    }
    pub fn build(
        &mut self,
        device: &ash::Device,
        shader_stages: ShaderStageFlags,
    ) -> DescriptorSetLayout {
        for binding in &mut self.bindings {
            binding.stage_flags |= shader_stages;
        }
        let create_info = DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);

        let layout = unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        };
        layout
    }
}
