use super::twm::*;
use std::env;

#[test]

fn import_and_export() {
    let temp_path = env::temp_dir()
        .join("twmnet_test.bin")
        .into_string()
        .unwrap();

    // export test
    let twm_net = TWMNet::new();
    assert!(twm_net.export(&temp_path).is_ok());

    // import test
    let twm_net = TWMNet::import(&temp_path);
    assert!(twm_net.is_ok() && !twm_net.unwrap().as_ptr().is_null());
}
