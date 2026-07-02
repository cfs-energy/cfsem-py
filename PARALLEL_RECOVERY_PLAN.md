# Parallel Matrix-Free Recovery Plan

## Goal

Add Rayon-backed parallelism to the solenoid stress FEM matrix-free recovery paths by chunking over point locations while preserving current numerical results, output ordering, and error behavior as much as practical.

This plan covers direct recovery only:

- `Structural2dModel::strain`
- `Structural2dModel::stress`
- `Structural2dModel::thermal_strain`
- `Structural2dModel::thermal_stress`

It does not cover matrix-free RHS assembly. RHS assembly accumulates many element or face contributions into shared reduced DOFs and needs per-worker RHS buffers plus a reduction, which is a separate design.

## Current State

Recovery is implemented in `src/physics/solenoid_stress/model.rs` with serial loops over located points:

- `evaluate_strain_for_locations_for_family`
- `evaluate_stress_for_locations_for_family`
- `evaluate_thermal_for_locations_for_family`

Each point computes one independent `[f64; 4]` output from:

- the owning `element_index`,
- that point's reference coordinates,
- mesh geometry,
- element material data,
- and either full displacements or nodal temperatures.

There is no cross-point accumulation. This makes recovery naturally parallel over point locations.

## Design

Parallelize over contiguous chunks of point indices, not over elements.

Reasoning:

- The public location inputs are ordered point arrays: `element_indices` and `reference_points`.
- Query points can be arbitrary, repeated, and not grouped by element.
- Chunking by point index preserves the public input/output order without a grouping and scatter step.
- For element-major quadrature locations, contiguous point chunks still align reasonably well with element order.

Use Rayon chunk iteration:

```rust
let chunk_size = chunksize(element_indices.len());
let chunks = ranges_for_len(element_indices.len(), chunk_size)
    .into_par_iter()
    .map(|(start, end)| {
        let mut out = Vec::with_capacity(end - start);
        for index in start..end {
            out.push(evaluate_one_point(index)?);
        }
        Ok(out)
    })
    .collect::<Result<Vec<_>, String>>()?;

Ok(chunks.into_iter().flatten().collect())
```

This keeps chunk output ordering deterministic because `collect::<Vec<_>>()` on an indexed Rayon iterator preserves source order.

## Implementation Steps

1. Extract single-point evaluators.

   Add private helpers for the repeated per-point bodies:

   - `evaluate_strain_at_location_for_family`
   - `evaluate_stress_at_location_for_family`
   - `evaluate_thermal_at_location_for_family`

   Each helper should take explicit borrowed inputs rather than `&self` where convenient:

   - mesh view,
   - material tables,
   - orientation angles,
   - formulation,
   - point element index,
   - reference point,
   - displacement or temperature input.

2. Add a small parallel collection helper.

   Add a private helper in `model.rs` such as:

   ```rust
   fn collect_location_chunks<T, F>(
       len: usize,
       evaluate_range: F,
   ) -> Result<Vec<T>, String>
   where
       T: Send,
       F: Fn(usize, usize) -> Result<Vec<T>, String> + Sync,
   ```

   It should:

   - return `Ok(Vec::new())` for `len == 0`,
   - use `ranges_for_len(len, chunksize(len))`,
   - run ranges with `into_par_iter()`,
   - collect `Result<Vec<Vec<T>>, String>`,
   - flatten chunks in order.

3. Decide when to parallelize.

   Use the existing model assembly flag:

   - if `self.assembly.par` is `true`, use the Rayon chunked path,
   - otherwise keep the serial path.

   Consider adding a small threshold so tiny point counts stay serial, for example `len < 2 * *PHYSICAL_CORES`. If `PHYSICAL_CORES` is private to `lib.rs`, either keep the first implementation simple without a threshold or add a small public-in-crate helper for the threshold. Avoid tuning complexity until benchmarks show a need.

4. Avoid capturing non-`Sync` model state accidentally.

   `Structural2dModel` contains an LU cache field. Even though recovery methods take `&self`, Rayon closures should not rely on `Structural2dModel: Sync`.

   Before entering the parallel closure, bind only the fields needed by recovery:

   - `let formulation = self.formulation;`
   - `let material_ids = &self.assembly.material_ids;`
   - `let material_table = &self.assembly.material_table;`
   - `let material_orientation_angles = self.assembly.material_orientation_angles.as_deref();`
   - `let thermal_material_table = self.assembly.thermal_material_table.as_ref();`
   - `let nodes = &self.analysis_nodes;`
   - local `elements` vector and mesh view.

   If the compiler still rejects shared references through the mesh view, pass `nodes` and `elements` separately to the single-point helper.

5. Preserve serial fallback behavior.

   Keep the existing serial loop, either directly or through a shared `evaluate_range` helper. This makes `par=False` behavior obvious and keeps debugging simple.

6. Add tests.

   Add Rust tests in `src/physics/solenoid_stress/model.rs` that compare serial and parallel models:

   - build the same small Quad4 model with `par=false` and `par=true`,
   - solve or supply a deterministic displacement vector,
   - query locations containing repeated and non-element-major `element_indices`,
   - compare `strain`, `stress`, `thermal_strain`, and `thermal_stress`.

   The test should assert exact equality where the operation order is unchanged within each point, or tight absolute tolerance otherwise. Since each point's arithmetic remains local and unchanged, exact equality should be expected.

7. Add a targeted benchmark or timing check if the repo already has a suitable benchmark pattern.

   Useful cases:

   - many arbitrary query points,
   - full quadrature recovery over a large mesh,
   - `par=false` vs `par=true`.

   Do not block implementation on a benchmark if there is no existing benchmark harness for structural recovery.

## Expected Code Touches

- `src/physics/solenoid_stress/model.rs`
  - import or reuse `chunksize` and `ranges_for_len`,
  - add helper for ordered parallel chunk collection,
  - refactor recovery loops into reusable range or single-point helpers,
  - switch recovery functions to serial or parallel based on `self.assembly.par`,
  - add tests.

No Python API change is required. The existing `par` flag on `assemble_structural_2d(...)` can control both sparse exports and matrix-free recovery.

## Error Handling

Rayon collection will return one `Err(String)` if any chunk fails. If multiple locations fail, the reported error may not always be the same as the serial first failing point.

If deterministic first-error behavior is required, each chunk can return `Result<Vec<T>, (usize, String)>`, where the index is the first failing point in that chunk. After collection, choose the smallest failing point index and return its message. That is more code, so start with ordinary `Result` unless tests or user-facing requirements demand exact serial error order.

## Verification

Run:

```bash
cargo test solenoid_stress
```

If the Python extension tests exercise recovery through the bindings, also run:

```bash
uv run pytest test/test_solenoid_stress_fem.py
```

## Non-Goals

- Parallelizing `build_rhs`.
- Changing sparse operator export behavior.
- Changing the Python API.
- Grouping query points by element.
- Changing numerical formulas or material rotation behavior.

