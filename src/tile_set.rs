use crate::{renderer::Texture2D, tile_map::TileType};
use serde_json::{json, Deserializer, Result, Serializer};

pub struct TileSet {
    available_tiles: Vec<Tile>,
}
pub struct Tile {
    texture_index: u32,
    tile_type: TileType,
}

impl TileSet {
    pub fn load_tileset(path: String) -> TileSet {
        
    }
}
