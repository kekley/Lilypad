use std::ffi::CString;

use ash::{
    util::read_spv,
    vk::{
        BlendFactor, BlendOp, ColorComponentFlags, CompareOp, CullModeFlags, DescriptorSetLayout,
        DescriptorSetLayoutBinding, DescriptorSetLayoutBindingFlagsCreateInfo,
        DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType, Extent2D,
        FrontFace, GraphicsPipelineCreateInfo, LogicOp, Pipeline, PipelineCache,
        PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
        PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PrimitiveTopology, PushConstantRange, Rect2D, RenderPass, SampleCountFlags, ShaderModule,
        ShaderModuleCreateFlags, ShaderModuleCreateInfo, ShaderStageFlags,
        VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, Viewport,
    },
};
use inline_spirv::include_spirv;

use crate::{shaders::Shader, vulkan_context::VulkanContext, HEIGHT, MSAA_SAMPLES, WIDTH};

use std::fmt::Debug;

pub trait VertexAttributes {
    fn input_attributes(&self) -> Vec<VertexInputAttributeDescription>;
    fn input_bindings(&self) -> Vec<VertexInputBindingDescription>;
}
pub trait AsPushConstantRange {
    fn push_constant_range(&self) -> PushConstantRange;
}

pub struct GraphicsPipelineBuilder<'a> {
    vertex_shader: Shader,
    fragment_shader: Shader,
    push_constants: Vec<PushConstantRange>,
    topology: PrimitiveTopology,
    descriptor_layouts: Vec<DescriptorSetLayout>,
    vertex_input_bindings: Vec<VertexInputBindingDescription>,
    vertex_input_attributes: Vec<VertexInputAttributeDescription>,
    multisampling_state: PipelineMultisampleStateCreateInfo<'a>,
    blend_attachment: PipelineColorBlendAttachmentState,
    depth_stencil_state: PipelineDepthStencilStateCreateInfo<'a>,
    viewport: Viewport,
    scissor: Rect2D,
}

impl GraphicsPipelineBuilder<'_> {
    pub fn new(
        vertex_shader: Shader,
        fragment_shader: Shader,
        vertex: Option<Box<dyn VertexAttributes>>,
        push_constant: Option<Box<dyn AsPushConstantRange>>,
    ) -> Self {
        let mut push_constant_range = Vec::new();
        match push_constant {
            Some(i) => push_constant_range.push(i.push_constant_range()),
            None => {}
        }

        let mut input_attributes = Vec::new();
        let mut input_bindings = Vec::new();

        match vertex {
            Some(v) => {
                input_bindings = v.input_bindings();
                input_attributes = v.input_attributes();
            }
            None => {}
        }
        Self {
            vertex_shader: vertex_shader,
            fragment_shader: fragment_shader,
            push_constants: push_constant_range,
            topology: PrimitiveTopology::TRIANGLE_LIST,
            descriptor_layouts: Vec::default(),
            vertex_input_attributes: input_attributes,
            vertex_input_bindings: input_bindings,
            multisampling_state: PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(SampleCountFlags::TYPE_1),
            blend_attachment: PipelineColorBlendAttachmentState::default()
                .color_write_mask(ColorComponentFlags::RGBA)
                .blend_enable(false),
            depth_stencil_state: PipelineDepthStencilStateCreateInfo::default()
                .depth_compare_op(CompareOp::NEVER)
                .max_depth_bounds(1.0),
            viewport: Viewport::default()
                .width(WIDTH as f32)
                .height(HEIGHT as f32)
                .min_depth(0.0)
                .max_depth(1.0),
            scissor: Rect2D::default().extent(Extent2D::default().width(WIDTH).height(HEIGHT)),
        }
    }

    pub fn add_descriptor_layout(&mut self, layout: DescriptorSetLayout) {
        self.descriptor_layouts.push(layout);
    }
    pub fn topology(&mut self, topology: PrimitiveTopology) {
        self.topology = topology
    }
    pub fn enable_msaa(&mut self) {
        self.multisampling_state = self.multisampling_state.sample_shading_enable(false);
        self.multisampling_state = self.multisampling_state.rasterization_samples(MSAA_SAMPLES);
    }
    pub fn enable_additive_blending(&mut self) {
        self.blend_attachment = self
            .blend_attachment
            .color_write_mask(ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD);
    }
    pub fn enable_alpha_blending(&mut self) {
        self.blend_attachment = self
            .blend_attachment
            .color_write_mask(ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD);
    }
    pub fn enable_depth_testing(&mut self, op: CompareOp) {
        self.depth_stencil_state = self
            .depth_stencil_state
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(op)
            .max_depth_bounds(1.0);
    }
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport = viewport;
    }
    pub fn set_scissor(&mut self, scissor: Rect2D) {
        self.scissor = scissor;
    }
    pub fn build(
        &self,
        device: &ash::Device,
        render_pass: RenderPass,
    ) -> (PipelineLayout, Pipeline) {
        let pipeline_layout_create_info = PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&self.push_constants)
            .set_layouts(&self.descriptor_layouts);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let stages = [
            self.vertex_shader.pipeline_stage_create_info(),
            self.fragment_shader.pipeline_stage_create_info(),
        ];

        let vertex_input = PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.vertex_input_bindings)
            .vertex_attribute_descriptions(&self.vertex_input_attributes);

        let input_assembly = PipelineInputAssemblyStateCreateInfo::default()
            .topology(self.topology)
            .primitive_restart_enable(false);
        let viewport = [self.viewport];
        let scissor = [self.scissor];
        let viewport_state = PipelineViewportStateCreateInfo::default()
            .viewports(&viewport)
            .scissors(&scissor);
        let rasterizer = PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(CullModeFlags::NONE)
            .front_face(FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let binding = [self.blend_attachment];
        let color_blending = PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&binding);

        let pipeline_create_info = GraphicsPipelineCreateInfo::default()
            .layout(pipeline_layout)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&self.multisampling_state)
            .color_blend_state(&color_blending)
            .stages(&stages)
            .render_pass(render_pass)
            .depth_stencil_state(&self.depth_stencil_state);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(PipelineCache::default(), &[pipeline_create_info], None)
                .unwrap()
        };
        (pipeline_layout, pipeline[0])
    }
}
