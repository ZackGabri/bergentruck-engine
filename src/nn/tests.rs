use std::env;

#[test]
fn twm_import_and_export_test() {
    let temp_path = env::temp_dir()
        .join("twmnet_test.bin")
        .into_string()
        .unwrap();

    // export test
    let twm_net = super::twm::TWMNet::new();
    assert!(twm_net.export(&temp_path).is_ok());

    // import test
    let twm_net = super::twm::TWMNet::import(&temp_path);
    assert!(twm_net.is_ok() && !twm_net.unwrap().as_ptr().is_null());
}
