use fidget::{
    gui::View3,
    mesh::{Octree, Settings},
    vm::VmShape,
};
use nalgebra::Vector3;

#[test]
fn test_octree_camera() {
    let (x, y, z) = fidget::context::Tree::axes();
    let sphere = ((x - 1.0).square() + (y - 1.0).square() + (z - 1.0).square())
        .sqrt()
        - 0.25;
    let shape = VmShape::from(sphere);

    let center = Vector3::new(1.0, 1.0, 1.0);
    let settings = Settings {
        depth: 4,
        world_to_model: View3::from_center_and_scale(center, 0.5)
            .world_to_model(),
        threads: None,
        ..Default::default()
    };

    let octree = Octree::build(&shape, &settings).unwrap().walk_dual();
    for v in octree.vertices.iter() {
        let n = (v - center).norm();
        assert!(n > 0.2 && n < 0.3, "invalid vertex at {v:?}: {n}");
    }
}
