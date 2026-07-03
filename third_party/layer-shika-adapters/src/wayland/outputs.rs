use layer_shika_domain::value_objects::output_handle::OutputHandle;
use std::collections::HashMap;
use wayland_client::backend::ObjectId;

pub(crate) use output_manager::{OutputManager, OutputManagerContext};
pub(crate) mod output_manager;

pub(crate) struct OutputMapping {
    object_to_handle: HashMap<ObjectId, OutputHandle>,
}

impl OutputMapping {
    pub fn new() -> Self {
        Self {
            object_to_handle: HashMap::new(),
        }
    }

    pub fn insert(&mut self, object_id: ObjectId) -> OutputHandle {
        let handle = OutputHandle::new();
        self.object_to_handle.insert(object_id, handle);
        handle
    }

    /// Insert a mapping for an output whose handle was already allocated elsewhere
    /// (e.g. by the `OutputManager` during hotplug registration), so both mappings
    /// agree on the same handle.
    pub fn insert_with_handle(&mut self, object_id: ObjectId, handle: OutputHandle) {
        self.object_to_handle.insert(object_id, handle);
    }

    pub fn remove_by_handle(&mut self, handle: OutputHandle) {
        self.object_to_handle.retain(|_, h| *h != handle);
    }

    pub fn get(&self, object_id: &ObjectId) -> Option<OutputHandle> {
        self.object_to_handle.get(object_id).copied()
    }

    pub fn remove(&mut self, object_id: &ObjectId) -> Option<OutputHandle> {
        self.object_to_handle.remove(object_id)
    }
}

impl Default for OutputMapping {
    fn default() -> Self {
        Self::new()
    }
}
