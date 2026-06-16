//! Deserialization logic for `ClickHouse` `Enum8` and `Enum16` types into Arrow
//! `DictionaryArray`.
//!
//! This module provides a function to deserialize `ClickHouse`’s native format for `Enum8` and
//! `Enum16` types into an Arrow `DictionaryArray` with integer keys (`Int8` or `Int16`) and
//! string values.
//!
//! The `deserialize` function reads raw indices (`i8` for `Enum8`, `i16` for `Enum16`,
//! little-endian) from the reader and maps each to its *position* in the enum declaration
//! (`pairs`), which becomes the Arrow `DictionaryArray` key. See the "Dictionary key space" note
//! below for why positions — not the raw `ClickHouse` indices — are used as keys.
//!
//! # Dictionary key space
//!
//! The Arrow dictionary's value array is built **once** from `pairs`, in `pairs` order, and the
//! Arrow key for each row is the *position of that row's `ClickHouse` index within `pairs`*. This
//! makes the key space deterministic and schema-defined (it depends only on the enum declaration,
//! not on which values happen to appear or in what order) — unlike a `StringDictionaryBuilder`,
//! which interns strings in first-seen order and so produces a data-dependent key space.
//!
//! Using *positions* (not the raw `ClickHouse` index) as keys also keeps the mapping total: CH enum
//! indices are signed and need not be contiguous or non-negative (e.g. `'x' = -5`), but positions
//! are always `0..pairs.len()`, which is exactly what an Arrow dictionary key must be.
//!
//! We build the array directly via `DictionaryArray::new` rather than driving a
//! `StringDictionaryBuilder`, mirroring the `LowCardinality` reader: the dictionary is already
//! known from the type, so there's no need to re-intern strings per row.
//!
//! # Examples
//! ```rust,ignore
//! use arrow::array::{ArrayRef, DictionaryArray, Int8Array, StringArray};
//! use arrow::datatypes::Int8Type;
//! use clickhouse_arrow::types::{Type, enums::deserialize, DeserializerState};
//! use std::sync::Arc;
//! use tokio::io::Cursor;
//!
//! #[tokio::test]
//! async fn test_enum8() {
//!     let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
//!     let data = vec![1, 2, 1]; // Keys: [1, 2, 1] -> ["a", "b", "a"]
//!     let mut reader = Cursor::new(data);
//!
//!     let array = deserialize(&Type::Enum8(pairs), &mut reader, 3, &[])
//!         .await
//!         .unwrap();
//!     let keys = Arc::new(Int8Array::from(vec![0, 1, 0])) as ArrayRef; // key = position in pairs
//!     let values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
//!     let expected =
//!         Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap()) as ArrayRef;
//!     assert_eq!(array.as_ref(), expected.as_ref());
//! }
//! ```
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{ArrowDictionaryKeyType, ArrowNativeType, Int8Type, Int16Type};
use tokio::io::AsyncReadExt;

use crate::arrow::builder::TypedBuilder;
use crate::io::ClickHouseRead;
use crate::{Error, Result, Type};

/// Deserializes a `ClickHouse` `Enum8` or `Enum16` type into an Arrow `DictionaryArray`.
///
/// The value array is built once from `pairs` (in declaration order) and the per-row Arrow key is
/// the *position* of that row's `ClickHouse` index within `pairs`. See [`build_enum_dict`] for the
/// key-space contract and the key-width ceiling. For example, an input of `[1, 2, 1]` for `Enum8`
/// with `pairs = [("a", 1), ("b", 2)]` produces a `DictionaryArray` with keys `[0, 1, 0]` and
/// values `["a", "b"]`, representing `["a", "b", "a"]`.
///
/// # Arguments
/// - `type_hint`: The `ClickHouse` `Type` (`Enum8` or `Enum16`) indicating the target type.
/// - `reader`: The async reader providing the `ClickHouse` native format data (raw `i8` or `i16`
///   indices).
/// - `rows`: The number of rows to deserialize.
/// - `nulls`: A slice indicating null values (`1` for null, `0` for non-null).
/// - `state`: A mutable `DeserializerState` for deserialization context (unused).
///
/// # Returns
/// A `Result` containing the deserialized `DictionaryArray` as an `ArrayRef` or a
/// `Error` if deserialization fails.
///
/// # Errors
/// - Returns `ArrowDeserialize` if:
///   - The `type_hint` is not `Enum8` or `Enum16`.
///   - An index is invalid (not found in `pairs`).
///   - The enum has more variants than the Arrow dictionary key type can index (see
///     [`build_enum_dict`]).
///   - The `DictionaryArray` construction fails.
/// - Returns `Io` if reading from the reader fails.
///
/// # Performance
/// - Reads the `rows` raw indices in one pass into a single `Vec`.
/// - Builds an index→position map once over `pairs` (small, typically <100 elements), then does an
///   O(1) lookup per row — no per-row scan and no per-row string interning.
/// - Constructs the value `StringArray` directly from `pairs`, sized to the dictionary, not `rows`.
pub(super) async fn deserialize_async<R: ClickHouseRead>(
    type_hint: &Type,
    builder: &mut TypedBuilder,
    reader: &mut R,
    rows: usize,
    nulls: &[u8],
) -> Result<ArrayRef> {
    match (type_hint, &*builder) {
        (Type::Enum8(pairs), TypedBuilder::Enum8(_)) => {
            let mut idx = vec![0_i8; rows];
            for slot in &mut idx {
                *slot = reader.read_i8().await?;
            }
            build_enum_dict::<Int8Type>(pairs, &idx, rows, nulls)
        }
        (Type::Enum16(pairs), TypedBuilder::Enum16(_)) => {
            let mut idx = vec![0_i16; rows];
            for slot in &mut idx {
                *slot = reader.read_i16_le().await?;
            }
            build_enum_dict::<Int16Type>(pairs, &idx, rows, nulls)
        }
        _ => {
            Err(Error::ArrowDeserialize(format!("Unexpected builder type for enum: {type_hint:?}")))
        }
    }
}

/// Build a `DictionaryArray` from the enum declaration and the raw wire indices.
///
/// The value array is the enum names in `pairs` order; the key for each row is the *position*
/// of that row's `ClickHouse` index within `pairs`. A wire index that is not declared in `pairs`
/// is a protocol error. Null rows (per `nulls`) become null keys.
///
/// # Key-width ceiling
///
/// The Arrow dictionary key type matches the enum's index width (`Enum8` → `Int8`, `Enum16` →
/// `Int16`), fixed by the type mapping in [`crate::arrow::types`]. Keys are *positions* in `pairs`,
/// so the largest representable enum is bounded by the **positive** range of the key type:
/// `Int8` positions max out at `i8::MAX` (127), `Int16` at `i16::MAX` (32767). `ClickHouse` itself
/// allows the full signed range (up to 256 `Enum8` / 65536 `Enum16` variants), so an enum with more
/// than `i8::MAX + 1` / `i16::MAX + 1` variants cannot be represented as this Arrow key type and is
/// rejected with `Error::ArrowDeserialize` rather than silently wrapping. Lifting this ceiling
/// would require widening the dictionary key in the type mapping (e.g. `Enum8` → `Int16` keys); out
/// of scope here.
fn build_enum_dict<K>(
    pairs: &[(String, K::Native)],
    indices: &[K::Native],
    rows: usize,
    nulls: &[u8],
) -> Result<ArrayRef>
where
    K: ArrowDictionaryKeyType,
    K::Native: std::hash::Hash + Eq + std::fmt::Display,
{
    use std::collections::HashMap;

    // index -> position in `pairs`. Built once; enum dictionaries are tiny.
    let mut pos_of: HashMap<K::Native, K::Native> = HashMap::with_capacity(pairs.len());
    for (position, (_, ch_index)) in pairs.iter().enumerate() {
        // The Arrow key is the *position* in `pairs`. It must fit the dictionary key type's
        // positive range (i8 -> 0..=127, i16 -> 0..=32767). CH permits more variants than that
        // (the index field is the full signed range), so an over-large enum is a clean error here,
        // not a silent wrap. See the "Key-width ceiling" note above.
        let key = K::Native::from_usize(position).ok_or_else(|| {
            Error::ArrowDeserialize(format!(
                "enum has too many variants ({}) to index with its Arrow dictionary key type; \
                 position {position} exceeds the key's positive range",
                pairs.len()
            ))
        })?;
        let _ = pos_of.insert(*ch_index, key);
    }

    let values: ArrayRef = Arc::new(StringArray::from_iter_values(pairs.iter().map(|(s, _)| s)));

    let keys: PrimitiveArray<K> = (0..rows)
        .map(|i| {
            if !nulls.is_empty() && nulls[i] != 0 {
                return Ok(None);
            }
            let ch_index = indices[i];
            pos_of.get(&ch_index).copied().map(Some).ok_or_else(|| {
                Error::ArrowDeserialize(format!(
                    "Invalid enum index: {ch_index} not found in pairs"
                ))
            })
        })
        .collect::<Result<Vec<Option<K::Native>>>>()?
        .into_iter()
        .collect();

    Ok(Arc::new(
        DictionaryArray::<K>::try_new(keys, values)
            .map_err(|e| Error::ArrowDeserialize(format!("enum dictionary construction: {e}")))?,
    ))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use arrow::array::{DictionaryArray, Int8Array, Int16Array, StringArray};
    use arrow::datatypes::{Int8Type, Int16Type};

    use super::*;

    // Helper to create a mock reader
    type MockReader = Cursor<Vec<u8>>;

    #[tokio::test]
    async fn test_deserialize_enum8() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let data = vec![1, 2, 1]; // Keys: [1, 2, 1] -> ["a", "b", "a"]
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum8(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int8),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let array = deserialize_async(&type_, &mut builder, &mut reader, 3, &[]).await.unwrap();
        let values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let expected = Arc::new(
            DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0, 1, 0]), values).unwrap(),
        ) as ArrayRef;
        assert_eq!(array.as_ref(), expected.as_ref());
    }

    #[tokio::test]
    async fn test_deserialize_enum16() {
        let pairs = vec![("x".to_string(), 10_i16), ("y".to_string(), 20_i16)];
        let data = vec![10, 0, 20, 0, 10, 0]; // Keys: [10, 20, 10] -> ["x", "y", "x"] in LE
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum16(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int16),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let array = deserialize_async(&type_, &mut builder, &mut reader, 3, &[]).await.unwrap();
        let values = Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef;
        let expected = Arc::new(
            DictionaryArray::<Int16Type>::try_new(Int16Array::from(vec![0, 1, 0]), values).unwrap(),
        ) as ArrayRef;
        assert_eq!(array.as_ref(), expected.as_ref());
    }

    #[tokio::test]
    async fn test_deserialize_enum8_nullable() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let data = vec![1, 2, 1]; // Keys: [1, 2, 1] -> ["a", null, "a"]
        let nulls = vec![0, 1, 0]; // Null bitmap: [non-null, null, non-null]
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum8(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int8),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let array = deserialize_async(&type_, &mut builder, &mut reader, 3, &nulls).await.unwrap();
        let values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let expected = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![Some(0), None, Some(0)]),
                values,
            )
            .unwrap(),
        ) as ArrayRef;
        assert_eq!(array.as_ref(), expected.as_ref());
    }

    #[tokio::test]
    async fn test_deserialize_enum8_empty() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let data = vec![]; // Empty
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum8(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int8),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let array = deserialize_async(&type_, &mut builder, &mut reader, 0, &[]).await.unwrap();
        let values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let expected = Arc::new(
            DictionaryArray::<Int8Type>::try_new(Int8Array::from(Vec::<i8>::new()), values)
                .unwrap(),
        ) as ArrayRef;
        assert_eq!(array.as_ref(), expected.as_ref());
    }

    #[tokio::test]
    async fn test_deserialize_enum8_invalid_index() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let data = vec![3]; // Invalid key: 3
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum8(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int8),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let result = deserialize_async(&type_, &mut builder, &mut reader, 1, &[]).await;
        assert!(matches!(
            result,
            Err(Error::ArrowDeserialize(msg))
            if msg.contains("Invalid")
        ));
    }

    #[tokio::test]
    async fn test_deserialize_invalid_type() {
        let data = vec![];
        let mut reader = MockReader::new(data);

        let type_ = Type::Int32;
        let data_type = arrow::datatypes::DataType::Int32;
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let result = deserialize_async(&type_, &mut builder, &mut reader, 0, &[]).await;
        assert!(matches!(
            result,
            Err(Error::ArrowDeserialize(msg))
            if msg.contains("Unexpected builder")
        ));
    }

    fn dict8(arr: &ArrayRef) -> &DictionaryArray<Int8Type> {
        arr.as_any().downcast_ref::<DictionaryArray<Int8Type>>().expect("Int8 dictionary")
    }

    #[tokio::test]
    async fn test_deserialize_enum8_sparse_negative_indices() {
        // CH enum indices are signed and need not be contiguous. The Arrow
        // key for each row is the *position in pairs*, never the raw CH index
        // — so a negative index (which can't be an Arrow key) works fine.
        // pairs: x=-5 (pos 0), y=10 (pos 1), z=0 (pos 2).
        let pairs =
            vec![("x".to_string(), -5_i8), ("y".to_string(), 10_i8), ("z".to_string(), 0_i8)];
        // wire indices: [10, -5, 0, 10] -> rows [y, x, z, y] -> keys [1, 0, 2, 1].
        #[expect(clippy::cast_sign_loss)]
        let data: Vec<u8> = [10_i8, -5, 0, 10].iter().map(|v| *v as u8).collect();
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum8(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int8),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let array = deserialize_async(&type_, &mut builder, &mut reader, 4, &[]).await.unwrap();

        let dict = dict8(&array);
        assert_eq!(dict.keys(), &Int8Array::from(vec![1, 0, 2, 1]), "keys are positions in pairs");
        // The value array is the names in pairs order, regardless of which appeared.
        let values = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(values, &StringArray::from(vec!["x", "y", "z"]));
    }

    #[tokio::test]
    async fn test_deserialize_enum8_too_many_variants_errors() {
        // CH allows up to 256 Enum8 variants, but the Arrow key is Int8, whose
        // positive range caps positions at 127. A 129-variant enum makes
        // position 128 unrepresentable -> a clean error, never a silent wrap.
        // pairs: name_i with CH index i, for i in 0..129.
        let pairs: Vec<(String, i8)> = (0..129)
            .map(|i| {
                #[expect(clippy::cast_possible_truncation)]
                // indices 0..=127 then -128 (wraps in i8) keep them distinct enough for the map;
                // distinctness isn't what's under test — the position ceiling is.
                let ch = i as i8;
                (format!("v{i}"), ch)
            })
            .collect();
        // One row is enough to enter the build; the error fires while building
        // the position map, before any row lookup.
        let data = vec![0_u8];
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum8(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int8),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let result = deserialize_async(&type_, &mut builder, &mut reader, 1, &[]).await;
        assert!(matches!(
            result,
            Err(Error::ArrowDeserialize(msg)) if msg.contains("too many variants")
        ));
    }

    #[tokio::test]
    async fn test_deserialize_enum8_key_space_is_schema_defined() {
        // Fidelity property: even when only the *last* declared value appears,
        // its key is its position in pairs (2), not 0. A first-seen-interning
        // builder would have keyed it 0 — that's the bug this guards against.
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8), ("c".to_string(), 3_i8)];
        let data = vec![3_u8, 3, 3]; // all "c"
        let mut reader = MockReader::new(data);

        let type_ = Type::Enum8(pairs);
        let data_type = arrow::datatypes::DataType::Dictionary(
            Box::new(arrow::datatypes::DataType::Int8),
            Box::new(arrow::datatypes::DataType::Utf8),
        );
        let mut builder = TypedBuilder::try_new(&type_, &data_type).unwrap();
        let array = deserialize_async(&type_, &mut builder, &mut reader, 3, &[]).await.unwrap();

        let dict = dict8(&array);
        assert_eq!(dict.keys(), &Int8Array::from(vec![2, 2, 2]), "key == position of 'c' in pairs");
        let values = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(values, &StringArray::from(vec!["a", "b", "c"]));
    }
}
