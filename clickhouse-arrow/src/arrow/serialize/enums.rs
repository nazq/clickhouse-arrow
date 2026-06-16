/// Serialization logic for `ClickHouse` `Enum8` and `Enum16` types from Arrow arrays.
///
/// This module provides functions to serialize Arrow `DictionaryArray`, `PrimitiveArray`, or
/// `StringArray`/`StringViewArray` into `ClickHouse`’s native format for `Enum8` and `Enum16`
/// types.
///
/// The `serialize` function dispatches to type-specific functions (`write_enum8_values`,
/// `write_enum16_values`) based on the `Type` variant (`Enum8` or `Enum16`). These functions
/// validate that the input array values match the provided enum pairs (e.g., `[("a", 1), ("b",
/// 2)]`) and write the corresponding raw indices (`i8` for `Enum8`, `i16` for `Enum16`,
/// little-endian) to the writer. Null values are written as `0`, with nullability handled by a
/// separate bitmap in `null.rs` for `Nullable(Enum)` types.
///
/// The implementation supports four input array types:
/// - `DictionaryArray`: Resolves each row through its dictionary *string* into the
///   `ClickHouse` enum index. The dictionary need not match `pairs` in order, length, or
///   coverage — reordered, subset, and extra-slot dictionaries all serialize correctly. Only a
///   dictionary value that is not a declared enum name is an error.
/// - `PrimitiveArray`: Validates that values exist in `pairs` and writes the indices directly.
/// - `StringArray`: Maps strings to indices.
/// - `StringViewArray`: Maps strings to indices.
///
/// The serialization is optimized for small enum `pairs` (typically <100 elements) with
/// efficient validation and minimal allocations (only a small `HashMap` for `StringArray`).
///
/// # Examples
/// ```rust,ignore
/// use arrow::array::{ArrayRef, DictionaryArray, Int8Array, StringArray};
/// use arrow::datatypes::{Field, Int8Type};
/// use clickhouse_arrow::types::{Type, enums::serialize, SerializerState};
/// use std::sync::Arc;
///
/// #[tokio::test]
/// async fn test_serialize_enum() {
///     let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
///     let keys = Int8Array::from(vec![0, 1, 0]);
///     let values = StringArray::from(vec!["a", "b"]);
///     let array = Arc::new(
///         DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap()
///     ) as ArrayRef;
///     let field = Field::new("", .clone(), false);
///     let mut writer = vec![];
///     serialize(&Type::Enum8(pairs), &mut writer, &array, array.data_type())
///         .await
///         .unwrap();
///     assert_eq!(writer, vec![1, 2, 1]); // Raw indices
/// }
/// ```
use arrow::array::*;
use arrow::datatypes::*;
use tokio::io::AsyncWriteExt;

use crate::io::{ClickHouseBytesWrite, ClickHouseWrite};
use crate::{Error, Result, Type};

/// Serializes an Arrow array to `ClickHouse`’s native format for `Enum8` or `Enum16` types.
///
/// Dispatches to `write_enum8_values` or `write_enum16_values` based on the `Type` variant,
/// handling `DictionaryArray`, `PrimitiveArray`, or `StringArray`. Validates that values match
/// the provided enum pairs and writes raw indices (`i8` for `Enum8`, `i16` for `Enum16`,
/// little-endian). Null values are written as `0`, with nullability handled by `null.rs` for
/// `Nullable(Enum)` types.
///
/// # Arguments
/// - `type_hint`: The `ClickHouse` `Type` (`Enum8` or `Enum16`) indicating the target type.
/// - `field`: The Arrow `Field` providing schema information (unused).
/// - `values`: The Arrow array containing the data to serialize.
/// - `writer`: The async writer to serialize to.
/// - `state`: A mutable `SerializerState` for serialization context (unused).
///
/// # Returns
/// A `Result` indicating success or a `Error` if serialization fails.
///
/// # Errors
/// - Returns `ArrowSerialize` if:
///   - The `type_hint` is not `Enum8` or `Enum16`.
///   - The input array type is unsupported (not `DictionaryArray`, `PrimitiveArray`, or
///     `StringArray`).
///   - A dictionary value, primitive value, or string is not a declared enum name/index.
///   - Dictionary keys are out of bounds for the dictionary's own value array.
/// - Returns `Io` if writing to the writer fails.
pub(super) async fn serialize_async<W: ClickHouseWrite>(
    type_hint: &Type,
    writer: &mut W,
    values: &ArrayRef,
) -> Result<()> {
    match type_hint.strip_null() {
        Type::Enum8(pairs) => write_enum8_values(values, writer, pairs).await?,
        Type::Enum16(pairs) => write_enum16_values(values, writer, pairs).await?,
        _ => {
            return Err(Error::ArrowSerialize(format!("Unsupported data type: {type_hint:?}")));
        }
    }

    Ok(())
}

pub(super) fn serialize<W: ClickHouseBytesWrite>(
    type_hint: &Type,
    writer: &mut W,
    values: &ArrayRef,
) -> Result<()> {
    match type_hint.strip_null() {
        Type::Enum8(pairs) => put_enum8_values(values, writer, pairs)?,
        Type::Enum16(pairs) => put_enum16_values(values, writer, pairs)?,
        _ => {
            return Err(Error::ArrowSerialize(format!("Unsupported data type: {type_hint:?}")));
        }
    }

    Ok(())
}

/// Macro to generate serialization functions for `Enum8` and `Enum16` types.
///
/// Generates functions that serialize `DictionaryArray`, `PrimitiveArray`, or `StringArray` to
/// `ClickHouse`’s `Enum8` or `Enum16` format. Validates that values match the provided enum pairs
/// and writes keys as `i8` or `i16`.
///
/// # Arguments
/// - `$name`: The function name (e.g., `write_enum8_values`).
/// - `enum $pt`: The primitive type (`i8` for `Enum8`, `i16` for `Enum16`).
/// - `$write_fn`: The writer method (e.g., `write_i8`, `write_i16_le`).
/// - `[$($kt:ty),*]`: The dictionary key types (e.g., `Int8Type`, `Int16Type`).
/// - `[$($at:ty),*]`: The primitive array types (e.g., `Int8Type`, `Int16Type`).
macro_rules! write_enum_values {
    // Enum8 and Enum16
    ($name:ident, enum $pt:ty, $write_fn:ident, [$($kt:ty),*], [$($at:ty),*], [$($st:ty),*]) => {
        /// Serializes an Arrow array to `ClickHouse`’s native format for enum types.
        ///
        /// Supports `DictionaryArray`, `PrimitiveArray`, `StringArray`, or `StringViewArray`,
        /// validating that values match the provided enum pairs. Writes keys as the specified
        /// primitive type (`i8` for `Enum8`, `i16` for `Enum16`, little-endian). Null values are
        /// written as `0`, with nullability handled by `null.rs` for `Nullable(Enum)` types.
        ///
        /// # Arguments
        /// - `column`: The Arrow array containing the data.
        /// - `writer`: The async writer to serialize to.
        /// - `enum_values`: The enum pairs mapping strings to values (e.g., `[("a", 1), ("b", 2)]`).
        ///
        /// # Returns
        /// A `Result` indicating success or a `Error` if serialization fails.
        ///
        /// # Errors
        /// - Returns `ArrowSerialize` if:
        ///   - The input array type is unsupported.
        ///   - A dictionary value, primitive value, or string is not a declared enum name/index.
        ///   - Dictionary keys are out of bounds for the dictionary's own value array.
        /// - Returns `Io` if writing to the writer fails.
        ///
        /// # Performance
        /// Optimized for small `enum_values` (typically <100 elements):
        /// - For `DictionaryArray`: builds a name→index map once, then resolves each dictionary
        ///   slot to its enum index once (dict-sized, not row-sized); per-row writes are O(1).
        /// - For `PrimitiveArray`: linear search over `enum_values`, cache-friendly.
        /// - For `StringArray`: a small `HashMap` lookup, O(m) allocation where m is `enum_values.len()`.
        #[allow(unused_comparisons)]
        #[allow(clippy::too_many_lines)]
        #[allow(clippy::cast_lossless)]
        #[allow(clippy::cast_sign_loss)]
        #[allow(clippy::cast_possible_wrap)]
        #[allow(clippy::cast_possible_truncation)]
        #[allow(trivial_numeric_casts)]
        async fn $name<W: ClickHouseWrite>(
            column: &::arrow::array::ArrayRef,
            writer: &mut W,
            enum_values: &[(String, $pt)], // From Type::Enum8 or Enum16
        ) -> Result<()> {
            // DictionaryArray case
            //
            // Resolve each row through its dictionary *string* into the enum's
            // CH index, rather than assuming the Arrow key space equals the
            // enum index space. This accepts any valid dictionary — reordered,
            // a subset of the enum, or with unused extra slots — not just one
            // whose value table is positionally identical to `pairs`.
            $(
                if let Some(array) = column.as_any().downcast_ref::<DictionaryArray<$kt>>() {
                    let keys = array.keys();
                    let values = array.values().as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                        Error::ArrowSerialize("Enum values must be strings".into())
                    })?;

                    // string -> CH enum index, built once.
                    let name_to_index: std::collections::HashMap<&str, $pt> = enum_values
                        .iter()
                        .map(|(s, v)| (s.as_str(), *v))
                        .collect();

                    // Arrow dictionary key -> CH enum index, resolved through the
                    // dictionary's string values, built once over the dictionary
                    // (size = dict, not rows).
                    let key_to_index: Vec<$pt> = (0..values.len())
                        .map(|d| {
                            let name = values.value(d);
                            name_to_index.get(name).copied().ok_or_else(|| {
                                Error::ArrowSerialize(format!(
                                    "Enum dictionary value '{name}' not in enum"
                                ))
                            })
                        })
                        .collect::<Result<Vec<$pt>>>()?;

                    for i in 0..keys.len() {
                        let value = if keys.is_null(i) {
                            0 // Null as 0 (Nullable handled by null.rs)
                        } else {
                            let key = keys.value(i);
                            if key < 0 || key as usize >= key_to_index.len() {
                                return Err(Error::ArrowSerialize(
                                    format!("Dictionary key {key} out of bounds")
                                ));
                            }
                            key_to_index[key as usize]
                        };
                        writer.$write_fn(value).await?;
                    }

                    return Ok(());
                }
            )*

            // PrimitiveArray case
            $(
                if let Some(array) = column.as_any().downcast_ref::<PrimitiveArray<$at>>() {
                    for i in 0..array.len() {
                        let value = if array.is_null(i) { 0 } else { array.value(i) as $pt };
                        // Validate value is in enum range
                        if !enum_values.iter().any(|(_, v)| *v == value) {
                            return Err(Error::ArrowSerialize(
                                format!("Value {value} not found in enum")
                            ));
                        }
                        writer.$write_fn(value).await?;
                    }
                    return Ok(());
                }
            )*

            $(
            if let Some(array) = column.as_string_opt::<$st>() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0).await?
                    } else {
                        let value = array.value(i);
                        let key = value_map.get(value).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key).await?;
                    }
                }
                return Ok(());
            }

            if let Some(array) = column.as_binary_opt::<$st>() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0).await?
                    } else {
                        let value = array.value(i);
                        let value_str = ::std::str::from_utf8(value)?;
                        let key = value_map.get(value_str).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value_str}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key).await?;
                    }
                }
                return Ok(());
            }
            )*

            // Views
            if let Some(array) = column.as_string_view_opt() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0).await?
                    } else {
                        let value = array.value(i);
                        let key = value_map.get(value).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key).await?;
                    }
                }
                return Ok(());
            }

            if let Some(array) = column.as_binary_view_opt() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0).await?
                    } else {
                        let value = array.value(i);
                        let value_str = ::std::str::from_utf8(value)?;
                        let key = value_map.get(value_str).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value_str}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key).await?;
                    }
                }
                return Ok(());
            }

            Err(Error::ArrowSerialize(format!(
                "Expected DictionaryArray, PrimitiveArray, StringArray, or BinaryArray, got {:?}",
                column.data_type()
            )))
        }
    };
}

macro_rules! put_enum_values {
    // Enum8 and Enum16
    ($name:ident, enum $pt:ty, $write_fn:ident, [$($kt:ty),*], [$($at:ty),*], [$($st:ty),*]) => {
        #[allow(unused_comparisons)]
        #[allow(clippy::too_many_lines)]
        #[allow(clippy::cast_lossless)]
        #[allow(clippy::cast_sign_loss)]
        #[allow(clippy::cast_possible_wrap)]
        #[allow(clippy::cast_possible_truncation)]
        #[allow(trivial_numeric_casts)]
        fn $name<W: $crate::io::ClickHouseBytesWrite>(
            column: &::arrow::array::ArrayRef,
            writer: &mut W,
            enum_values: &[(String, $pt)], // From Type::Enum8 or Enum16
        ) -> Result<()> {
            // DictionaryArray case
            //
            // Resolve each row through its dictionary *string* into the enum's
            // CH index, rather than assuming the Arrow key space equals the
            // enum index space. This accepts any valid dictionary — reordered,
            // a subset of the enum, or with unused extra slots.
            $(
                if let Some(array) = column.as_any().downcast_ref::<DictionaryArray<$kt>>() {
                    let keys = array.keys();
                    let values = array.values().as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                        Error::ArrowSerialize("Enum values must be strings".into())
                    })?;

                    let name_to_index: std::collections::HashMap<&str, $pt> = enum_values
                        .iter()
                        .map(|(s, v)| (s.as_str(), *v))
                        .collect();

                    let key_to_index: Vec<$pt> = (0..values.len())
                        .map(|d| {
                            let name = values.value(d);
                            name_to_index.get(name).copied().ok_or_else(|| {
                                Error::ArrowSerialize(format!(
                                    "Enum dictionary value '{name}' not in enum"
                                ))
                            })
                        })
                        .collect::<Result<Vec<$pt>>>()?;

                    for i in 0..keys.len() {
                        let value = if keys.is_null(i) {
                            0 // Null as 0 (Nullable handled by null.rs)
                        } else {
                            let key = keys.value(i);
                            if key < 0 || key as usize >= key_to_index.len() {
                                return Err(Error::ArrowSerialize(
                                    format!("Dictionary key {key} out of bounds")
                                ));
                            }
                            key_to_index[key as usize]
                        };
                        writer.$write_fn(value);
                    }

                    return Ok(());
                }
            )*

            // PrimitiveArray case
            $(
                if let Some(array) = column.as_any().downcast_ref::<PrimitiveArray<$at>>() {
                    for i in 0..array.len() {
                        let value = if array.is_null(i) { 0 } else { array.value(i) as $pt };
                        // Validate value is in enum range
                        if !enum_values.iter().any(|(_, v)| *v == value) {
                            return Err(Error::ArrowSerialize(
                                format!("Value {value} not found in enum")
                            ));
                        }
                        writer.$write_fn(value);
                    }
                    return Ok(());
                }
            )*

            $(
            if let Some(array) = column.as_string_opt::<$st>() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0)
                    } else {
                        let value = array.value(i);
                        let key = value_map.get(value).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key);
                    }
                }
                return Ok(());
            }

            if let Some(array) = column.as_binary_opt::<$st>() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0)
                    } else {
                        let value = array.value(i);
                        let value_str = ::std::str::from_utf8(value)?;
                        let key = value_map.get(value_str).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value_str}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key);
                    }
                }
                return Ok(());
            }
            )*

            // Views
            if let Some(array) = column.as_string_view_opt() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0)
                    } else {
                        let value = array.value(i);
                        let key = value_map.get(value).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key);
                    }
                }
                return Ok(());
            }

            if let Some(array) = column.as_binary_view_opt() {
                let value_map: std::collections::HashMap<&str, $pt> = enum_values
                    .iter()
                    .map(|(s, v)| (s.as_str(), *v))
                    .collect();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        writer.$write_fn(0)
                    } else {
                        let value = array.value(i);
                        let value_str = ::std::str::from_utf8(value)?;
                        let key = value_map.get(value_str).copied().ok_or(
                            Error::ArrowSerialize(format!(
                                "String '{value_str}' not in enum"
                            ))
                        )?;
                        writer.$write_fn(key);
                    }
                }
                return Ok(());
            }

            Err(Error::ArrowSerialize(format!(
                "Expected DictionaryArray, PrimitiveArray, StringArray, or BinaryArray, got {:?}",
                column.data_type()
            )))
        }
    };
}

write_enum_values!(
    write_enum8_values,
    enum i8,
    write_i8,
    [Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [i32, i64]
);
write_enum_values!(
    write_enum16_values,
    enum i16,
    write_i16_le,
    [Int16Type, Int8Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [Int16Type, Int8Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [i32, i64]
);

put_enum_values!(
    put_enum8_values,
    enum i8,
    put_i8,
    [Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [i32, i64]
);
put_enum_values!(
    put_enum16_values,
    enum i16,
    put_i16_le,
    [Int16Type, Int8Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [Int16Type, Int8Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type],
    [i32, i64]
);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{DictionaryArray, Int8Array, Int16Array, StringArray};
    use arrow::datatypes::{Int8Type, Int16Type};

    use super::*;

    type MockWriter = Vec<u8>;

    #[tokio::test]
    async fn test_serialize_enum8_dictionary() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["a", "b"]);
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![1, 2, 1]);
    }

    #[tokio::test]
    async fn test_serialize_enum8_primitive() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(Int8Array::from(vec![1, 2, 1])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![1, 2, 1]);
    }

    #[tokio::test]
    async fn test_serialize_enum8_string() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(StringArray::from(vec!["a", "b", "a"])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![1, 2, 1]);
    }

    #[tokio::test]
    async fn test_serialize_enum8_nullable() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(StringArray::from(vec![Some("a"), None, Some("a")])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![1, 0, 1]);
    }

    #[tokio::test]
    async fn test_serialize_enum16_dictionary() {
        let pairs = vec![("x".to_string(), 10_i16), ("y".to_string(), 20_i16)];
        let keys = Int16Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["x", "y"]);
        let array = Arc::new(DictionaryArray::<Int16Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum16(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![10, 0, 20, 0, 10, 0]); // Little-endian
    }

    #[tokio::test]
    async fn test_serialize_enum8_empty() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(Int8Array::from(Vec::<i8>::new())) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert!(writer.is_empty());
    }

    #[tokio::test]
    async fn test_serialize_enum8_invalid_value() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(Int8Array::from(vec![3])) as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize_async(&Type::Enum8(pairs), &mut writer, &array).await;
        assert!(matches!(
            result,
            Err(Error::ArrowSerialize(msg))
            if msg.contains("Value 3 not found in enum")
        ));
    }

    #[tokio::test]
    async fn test_serialize_enum8_dictionary_invalid_array() {
        let array = Arc::new(TimestampSecondArray::from(Vec::<i64>::new())) as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize_async(&Type::Enum8(vec![]), &mut writer, &array).await;
        assert!(matches!(result, Err(Error::ArrowSerialize(_))));
    }

    #[tokio::test]
    async fn test_serialize_enum8_dictionary_invalid_value() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(Vec::<i8>::new());
        let values = TimestampSecondArray::from(Vec::<i64>::new());
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize_async(&Type::Enum8(pairs), &mut writer, &array).await;
        assert!(matches!(result, Err(Error::ArrowSerialize(_))));
    }

    #[tokio::test]
    async fn test_serialize_enum8_dictionary_value_not_in_enum() {
        // A dictionary slot holds "c", which is not a declared enum name. Even
        // though it's only referenced indirectly, the dictionary is resolved
        // up front, so this is a hard error. (Previously this was a "length
        // mismatch" check; the length check is gone — the rejection now comes
        // from the unknown value.)
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["a", "b", "c"]);
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize_async(&Type::Enum8(pairs), &mut writer, &array).await;
        assert!(matches!(
            result,
            Err(Error::ArrowSerialize(msg)) if msg.contains("not in enum")
        ));
    }

    #[tokio::test]
    async fn test_serialize_enum8_dictionary_superset_all_valid() {
        // A dictionary that is a *superset* of the rows' values, where every
        // slot is a valid enum name (here the enum has 3 names and the dict
        // lists all 3, but the rows only reference two). This must serialize —
        // there is no length-equality requirement.
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8), ("c".to_string(), 3_i8)];
        // dict ["a","b","c"]; rows reference only a (->1) and c (->3).
        let keys = Int8Array::from(vec![0, 2, 0]);
        let values = StringArray::from(vec!["a", "b", "c"]);
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![1, 3, 1]);
    }

    #[tokio::test]
    async fn test_serialize_enum16_uint_type_ok() {
        let pairs = vec![("x".to_string(), 10_i16), ("y".to_string(), 20_i16)];
        let array = Arc::new(UInt8Array::from(vec![10])) as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize_async(&Type::Enum16(pairs), &mut writer, &array).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_serialize_enum8_negative_values() {
        let pairs = vec![("neg".to_string(), -1_i8), ("pos".to_string(), 1_i8)];
        let array = Arc::new(Int8Array::from(vec![-1, 1, -1])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![255, 1, 255]); // -1 as i8 = 255
    }

    #[tokio::test]
    async fn test_serialize_enum16_sparse_values() {
        let pairs = vec![("a".to_string(), 100_i16), ("b".to_string(), 200_i16)];
        let array = Arc::new(Int16Array::from(vec![100, 200, 100])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum16(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![100, 0, 200, 0, 100, 0]); // Little-endian
    }

    #[tokio::test]
    async fn test_serialize_enum8_dictionary_reordered() {
        // Dictionary values in a different order than `pairs`. Rows are
        // ["b", "a", "b"]; resolving through the dictionary string gives CH
        // indices [2, 1, 2]. The serializer must accept this, not reject it.
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["b", "a"]); // reordered vs pairs
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![2, 1, 2]);
    }

    #[tokio::test]
    async fn test_serialize_enum8_dictionary_subset() {
        // Dictionary holds only one of the enum's values. Rows are all "b" =>
        // CH index 2. A subset dictionary is valid and must serialize.
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 0, 0]);
        let values = StringArray::from(vec!["b"]); // subset of the enum
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize_async(&Type::Enum8(pairs), &mut writer, &array).await.unwrap();
        assert_eq!(writer, vec![2, 2, 2]);
    }

    #[tokio::test]
    async fn test_serialize_enum8_dictionary_unknown_value_errors() {
        // A dictionary value that isn't in the enum is still a hard error.
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1]);
        let values = StringArray::from(vec!["a", "zzz"]); // "zzz" not in enum
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize_async(&Type::Enum8(pairs), &mut writer, &array).await;
        assert!(matches!(
            result,
            Err(Error::ArrowSerialize(msg)) if msg.contains("not in enum")
        ));
    }

    #[tokio::test]
    async fn test_write_enum8_string_like_values() {
        let cases = vec![
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
            Arc::new(BinaryArray::from_opt_vec(vec![Some(b"a"), Some(b"b"), None])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![Some(b"a" as &[u8]), Some(b"b"), None]))
                as ArrayRef,
            Arc::new(LargeBinaryArray::from_opt_vec(vec![Some(b"a"), Some(b"b"), None]))
                as ArrayRef,
        ];
        let enum_values = vec![("a".to_string(), 1), ("b".to_string(), 2)];
        for array in cases {
            let mut writer = MockWriter::new();
            serialize_async(&Type::Enum8(enum_values.clone()), &mut writer, &array).await.unwrap();
            assert_eq!(writer, vec![1, 2, 0]);
        }
    }

    #[tokio::test]
    async fn test_serialize_enum_wrong_type() {
        let mut writer = MockWriter::new();
        let result = serialize_async(
            &Type::String,
            &mut writer,
            &(Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef),
        )
        .await;
        assert!(matches!(result, Err(Error::ArrowSerialize(_))));
    }
}

#[cfg(test)]
mod tests_sync {
    use std::sync::Arc;

    use arrow::array::{DictionaryArray, Int8Array, Int16Array, StringArray};
    use arrow::datatypes::{Int8Type, Int16Type};

    use super::*;

    type MockWriter = Vec<u8>;

    #[test]
    fn test_serialize_enum8_dictionary() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["a", "b"]);
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![1, 2, 1]);
    }

    #[test]
    fn test_serialize_enum8_primitive() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(Int8Array::from(vec![1, 2, 1])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![1, 2, 1]);
    }

    #[test]
    fn test_serialize_enum8_string() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(StringArray::from(vec!["a", "b", "a"])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![1, 2, 1]);
    }

    #[test]
    fn test_serialize_enum8_nullable() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(StringArray::from(vec![Some("a"), None, Some("a")])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![1, 0, 1]);
    }

    #[test]
    fn test_serialize_enum16_dictionary() {
        let pairs = vec![("x".to_string(), 10_i16), ("y".to_string(), 20_i16)];
        let keys = Int16Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["x", "y"]);
        let array = Arc::new(DictionaryArray::<Int16Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum16(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![10, 0, 20, 0, 10, 0]); // Little-endian
    }

    #[test]
    fn test_serialize_enum8_empty() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(Int8Array::from(Vec::<i8>::new())) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert!(writer.is_empty());
    }

    #[test]
    fn test_serialize_enum8_invalid_value() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let array = Arc::new(Int8Array::from(vec![3])) as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize(&Type::Enum8(pairs), &mut writer, &array);
        assert!(matches!(
            result,
            Err(Error::ArrowSerialize(msg))
            if msg.contains("Value 3 not found in enum")
        ));
    }

    #[test]
    fn test_serialize_enum8_dictionary_invalid_array() {
        let array = Arc::new(TimestampSecondArray::from(Vec::<i64>::new())) as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize(&Type::Enum8(vec![]), &mut writer, &array);
        assert!(matches!(result, Err(Error::ArrowSerialize(_))));
    }

    #[test]
    fn test_serialize_enum8_dictionary_invalid_value() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(Vec::<i8>::new());
        let values = TimestampSecondArray::from(Vec::<i64>::new());
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize(&Type::Enum8(pairs), &mut writer, &array);
        assert!(matches!(result, Err(Error::ArrowSerialize(_))));
    }

    #[test]
    fn test_serialize_enum8_dictionary_value_not_in_enum() {
        // See the async twin: "c" is not a declared enum name -> hard error.
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["a", "b", "c"]);
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize(&Type::Enum8(pairs), &mut writer, &array);
        assert!(matches!(
            result,
            Err(Error::ArrowSerialize(msg)) if msg.contains("not in enum")
        ));
    }

    #[test]
    fn test_serialize_enum8_dictionary_superset_all_valid() {
        // Superset dictionary, all slots valid enum names; rows reference a
        // subset. Must serialize (no length-equality requirement).
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8), ("c".to_string(), 3_i8)];
        let keys = Int8Array::from(vec![0, 2, 0]);
        let values = StringArray::from(vec!["a", "b", "c"]);
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![1, 3, 1]);
    }

    #[test]
    fn test_serialize_enum16_uint_type_ok() {
        let pairs = vec![("x".to_string(), 10_i16), ("y".to_string(), 20_i16)];
        let array = Arc::new(UInt8Array::from(vec![10])) as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize(&Type::Enum16(pairs), &mut writer, &array);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialize_enum8_negative_values() {
        let pairs = vec![("neg".to_string(), -1_i8), ("pos".to_string(), 1_i8)];
        let array = Arc::new(Int8Array::from(vec![-1, 1, -1])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![255, 1, 255]); // -1 as i8 = 255
    }

    #[test]
    fn test_serialize_enum16_sparse_values() {
        let pairs = vec![("a".to_string(), 100_i16), ("b".to_string(), 200_i16)];
        let array = Arc::new(Int16Array::from(vec![100, 200, 100])) as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum16(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![100, 0, 200, 0, 100, 0]); // Little-endian
    }

    #[test]
    fn test_serialize_enum8_dictionary_reordered() {
        // See the async twin: ["b","a","b"] -> CH indices [2,1,2].
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec!["b", "a"]); // reordered vs pairs
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        serialize(&Type::Enum8(pairs), &mut writer, &array).unwrap();
        assert_eq!(writer, vec![2, 1, 2]);
    }

    #[test]
    fn test_serialize_enum8_dictionary_unknown_value_errors() {
        let pairs = vec![("a".to_string(), 1_i8), ("b".to_string(), 2_i8)];
        let keys = Int8Array::from(vec![0, 1]);
        let values = StringArray::from(vec!["a", "zzz"]); // "zzz" not in enum
        let array = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
            as ArrayRef;
        let mut writer = MockWriter::new();
        let result = serialize(&Type::Enum8(pairs), &mut writer, &array);
        assert!(matches!(
            result,
            Err(Error::ArrowSerialize(msg)) if msg.contains("not in enum")
        ));
    }

    #[test]
    fn test_write_enum8_string_like_values() {
        let cases = vec![
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
            Arc::new(BinaryArray::from_opt_vec(vec![Some(b"a"), Some(b"b"), None])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![Some(b"a" as &[u8]), Some(b"b"), None]))
                as ArrayRef,
            Arc::new(LargeBinaryArray::from_opt_vec(vec![Some(b"a"), Some(b"b"), None]))
                as ArrayRef,
        ];
        let enum_values = vec![("a".to_string(), 1), ("b".to_string(), 2)];
        for array in cases {
            let mut writer = MockWriter::new();
            serialize(&Type::Enum8(enum_values.clone()), &mut writer, &array).unwrap();
            assert_eq!(writer, vec![1, 2, 0]);
        }
    }

    #[test]
    fn test_serialize_enum_wrong_type() {
        let mut writer = MockWriter::new();
        let result = serialize(
            &Type::String,
            &mut writer,
            &(Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef),
        );
        assert!(matches!(result, Err(Error::ArrowSerialize(_))));
    }
}
