use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapEntry {
    pub count: u64,
    pub item: u64,
}

pub struct IndexedMinHeap {
    heap: Vec<HeapEntry>,
    position: HashMap<u64, usize>,
}

impl IndexedMinHeap {
    pub fn new() -> Self {
        Self { heap: Vec::new(), position: HashMap::new() }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn contains(&self, item: u64) -> bool {
        self.position.contains_key(&item)
    }

    pub fn count_of(&self, item: u64) -> Option<u64> {
        self.position.get(&item).map(|&idx| self.heap[idx].count)
    }

    pub fn peek_min(&self) -> Option<HeapEntry> {
        self.heap.first().copied()
    }

    pub fn push(&mut self, item: u64, count: u64) {
        debug_assert!(!self.position.contains_key(&item));
        let idx = self.heap.len();
        self.heap.push(HeapEntry { count, item });
        self.position.insert(item, idx);
        self.sift_up(idx);
    }

    pub fn set_count(&mut self, item: u64, new_count: u64) {
        let idx = *self.position.get(&item).expect("item not present in heap");
        let old_count = self.heap[idx].count;
        self.heap[idx].count = new_count;
        if new_count < old_count {
            self.sift_up(idx);
        } else if new_count > old_count {
            self.sift_down(idx);
        }
    }

    pub fn replace_min(&mut self, new_item: u64, new_count: u64) -> HeapEntry {
        let old = self.heap[0];
        self.position.remove(&old.item);
        self.heap[0] = HeapEntry { count: new_count, item: new_item };
        self.position.insert(new_item, 0);
        self.sift_down(0);
        old
    }

    pub fn remove(&mut self, item: u64) -> Option<u64> {
        let idx = *self.position.get(&item)?;
        let old_count = self.heap[idx].count;
        let last = self.heap.len() - 1;
        self.swap(idx, last);
        self.heap.pop();
        self.position.remove(&item);
        if idx < self.heap.len() {
            self.sift_down(idx);
            self.sift_up(idx);
        }
        Some(old_count)
    }

    pub fn pop_min(&mut self) -> Option<HeapEntry> {
        if self.heap.is_empty() {
            return None;
        }
        let min = self.heap[0];
        self.remove(min.item);
        Some(min)
    }

    pub fn entries(&self) -> impl Iterator<Item = &HeapEntry> {
        self.heap.iter()
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.position.insert(self.heap[i].item, i);
        self.position.insert(self.heap[j].item, j);
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.heap[idx].count < self.heap[parent].count {
                self.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.heap.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;
            if left < len && self.heap[left].count < self.heap[smallest].count {
                smallest = left;
            }
            if right < len && self.heap[right].count < self.heap[smallest].count {
                smallest = right;
            }
            if smallest == idx {
                break;
            }
            self.swap(idx, smallest);
            idx = smallest;
        }
    }
}

impl Default for IndexedMinHeap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_peek_min_tracks_smallest_count() {
        let mut heap = IndexedMinHeap::new();
        heap.push(10, 5);
        heap.push(20, 2);
        heap.push(30, 8);
        assert_eq!(heap.peek_min(), Some(HeapEntry { count: 2, item: 20 }));
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn set_count_resifts_up_and_down_correctly() {
        let mut heap = IndexedMinHeap::new();
        heap.push(1, 10);
        heap.push(2, 20);
        heap.push(3, 30);
        heap.set_count(3, 1);
        assert_eq!(heap.peek_min(), Some(HeapEntry { count: 1, item: 3 }));
        heap.set_count(3, 100);
        assert_ne!(heap.peek_min().unwrap().item, 3);
    }

    #[test]
    fn remove_by_key_maintains_heap_invariant() {
        let mut heap = IndexedMinHeap::new();
        for (item, count) in [(1u64, 5u64), (2, 3), (3, 8), (4, 1), (5, 9)] {
            heap.push(item, count);
        }
        let removed = heap.remove(4);
        assert_eq!(removed, Some(1));
        assert_eq!(heap.len(), 4);
        assert!(!heap.contains(4));

        let mut prev = 0u64;
        while let Some(entry) = heap.pop_min() {
            assert!(entry.count >= prev, "heap invariant violated: {} < {}", entry.count, prev);
            prev = entry.count;
        }
    }

    #[test]
    fn replace_min_swaps_in_new_item_at_root() {
        let mut heap = IndexedMinHeap::new();
        heap.push(1, 5);
        heap.push(2, 2);
        heap.push(3, 8);
        let old = heap.replace_min(99, 100);
        assert_eq!(old, HeapEntry { count: 2, item: 2 });
        assert!(!heap.contains(2));
        assert!(heap.contains(99));
        assert_eq!(heap.peek_min(), Some(HeapEntry { count: 5, item: 1 }));
    }
}
