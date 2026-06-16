#![allow(unused_crate_dependencies)]

pub mod common;
pub mod tests;

const TRACING_DIRECTIVES: &[(&str, &str)] =
    &[("testcontainers", "debug"), ("clickhouse_arrow", "debug")];

// Cross-client primitive round-trip suite.
//
// One direction: raw SQL INSERT (server parses literals) -> arrow read.
// Other direction: arrow RecordBatch INSERT -> SQL `SELECT toString(col)` read.
// Both must agree with the ClickHouse server.
#[cfg(feature = "test-utils")]
e2e_test!(
    e2e_cross_client_primitives,
    tests::cross_client::test_cross_client_primitives,
    TRACING_DIRECTIVES,
    None
);

// Sparse default-fill: a non-nullable Decimal/DateTime64 column dominated by
// defaults goes sparse on the wire; the omitted rows must reconstruct as the
// type zero, not NULL.
#[cfg(feature = "test-utils")]
e2e_test!(
    e2e_cross_client_sparse_default_fill,
    tests::cross_client::test_sparse_default_fill_non_nullable,
    TRACING_DIRECTIVES,
    None
);

// Enum dictionary encoding: SQL->Arrow asserts the dictionary key space is
// the enum declaration order; Arrow->SQL inserts a reordered dictionary to
// prove the writer resolves by string, not by assuming key == enum index.
#[cfg(feature = "test-utils")]
e2e_test!(
    e2e_cross_client_enum_dictionary,
    tests::cross_client::test_enum_dictionary,
    TRACING_DIRECTIVES,
    None
);
