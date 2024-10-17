use std::{
    char,
    collections::{HashMap, HashSet},
    fs::File,
    sync::Arc,
};

use ash::vk::{
    BufferUsageFlags, DescriptorType, Filter, Format, ImageAspectFlags, ImageTiling,
    ImageUsageFlags, Pipeline, PipelineLayout, SampleCountFlags, Sampler, SamplerCreateInfo,
    ShaderStageFlags, VertexInputAttributeDescription, VertexInputBindingDescription,
    VertexInputRate,
};
use freetype::face::LoadFlag;
use freetype::Library;
use glam::{I16Vec2, IVec2, Vec2, Vec3, Vec4};

use crate::{
    descriptors::DescriptorAllocator,
    pipeline_builder::{self, GraphicsPipelineBuilder, VertexAttributes},
    rects::Extent,
    renderer::{RenderContext, Texture2D},
    resource_manager::{self, AllocatedBuffer, AllocatedImage, ResourceManager},
    shaders::{
        DescriptorLayoutBuilder, Shader, FONT_FRAGMENT_SHADER_CODE, FONT_VERTEX_SHADER_CODE,
    },
};
struct Character {
    size: IVec2,
    bearing: IVec2,
    advance: u32,
}

//fn create_pipeline() -> (PipelineLayout, Pipeline) {}
fn build_font_data(
    font_size: u32,
    resource_manager: &ResourceManager,
) -> (HashMap<char, Character>, Vec<AllocatedImage>) {
    let lib = Library::init().unwrap();
    let face = lib
        .new_face("./assets/fonts/plus-1p-regular.ttf", 0)
        .unwrap();

    face.set_char_size(0, (font_size * 64).try_into().unwrap(), 1920, 1080)
        .unwrap();

    let mut characters: HashMap<char, Character> = HashMap::with_capacity(128);
    let mut images = Vec::with_capacity(128);
    (0..128).for_each(|i| {
        face.load_char(i, LoadFlag::DEFAULT).unwrap();
        face.glyph()
            .render_glyph(freetype::RenderMode::Sdf)
            .unwrap();

        let width = face.glyph().bitmap().width();
        let height = face.glyph().bitmap().rows();

        let sdf = resource_manager.create_image_from_data(
            Extent::new(width as u32, height as u32),
            1,
            SampleCountFlags::TYPE_1,
            Format::R8_UINT,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::SAMPLED,
            ImageAspectFlags::COLOR,
            face.glyph().bitmap().buffer().as_ptr(),
            face.glyph().bitmap().buffer().len(),
        );
        images.push(sdf);

        let character = Character {
            size: IVec2::new(face.glyph().bitmap().width(), face.glyph().bitmap().rows()),
            bearing: IVec2::new(face.glyph().bitmap_left(), face.glyph().bitmap_top()),
            advance: face.glyph().advance().x as u32,
        };
        characters.insert(char::from_u32(i as u32).unwrap(), character);
    });
    (characters, images)
}
