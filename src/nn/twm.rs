use std::ffi::CString;

mod twm {
    #![allow(unused, nonstandard_style, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/TWMNet.rs"));
}

const INPUT_SIZE: u64 = 64;
const LAYER1_SIZE: u64 = 1024;
const LAYER2_SIZE: u64 = 256;
const LAYER3_SIZE: u64 = 256;
const OUTPUT_SIZE: u64 = 1;

pub type NetworkInput = [f64; INPUT_SIZE as usize];
pub type NetworkOutput = [f64; OUTPUT_SIZE as usize];

pub struct TWMNet {
    network_ptr: twm::PTWM_RNN_NETWORK,
}

impl TWMNet {
    pub fn new() -> Self {
        unsafe {
            let layer1 = twm::TWMNetworkRNNCreateLayer(LAYER1_SIZE, INPUT_SIZE, -1.0, 1.0, 0.01);
            let layer2 = twm::TWMNetworkRNNCreateLayer(LAYER2_SIZE, LAYER1_SIZE, -1.0, 1.0, 0.01);
            let layer3 = twm::TWMNetworkRNNCreateLayer(LAYER3_SIZE, LAYER2_SIZE, -1.0, 1.0, 0.01);
            let layer4 = twm::TWMNetworkRNNCreateLayer(OUTPUT_SIZE, LAYER3_SIZE, -1.0, 1.0, 0.01);

            let network = twm::TWMNetworkRNNCreateNetwork(4, layer1, layer2, layer3, layer4);

            Self {
                network_ptr: network,
            }
        }
    }

    pub fn import(path: &str) -> anyhow::Result<Self> {
        let path = std::path::absolute(path).unwrap();
        let c_path = CString::new(path.to_str().expect("path contains invalid UTF-8")).unwrap();
        let path_ptr = c_path.as_ptr() as *mut _;

        unsafe {
            let network = twm::TWMNetworkRNNImportNetwork(path_ptr);
            if network.is_null() {
                anyhow::bail!("Import failed");
            } else {
                Ok(Self {
                    network_ptr: network,
                })
            }
        }
    }

    pub fn export(&self, path: &str) -> anyhow::Result<()> {
        let path = std::path::absolute(path).unwrap();
        let c_path = CString::new(path.to_str().expect("path contains invalid UTF-8")).unwrap();
        let path_ptr = c_path.as_ptr() as *mut _;

        unsafe {
            if twm::TWMNetworkRNNExportNetwork(self.network_ptr, path_ptr) != 0 {
                anyhow::bail!("Export failed");
            }
        }

        Ok(())
    }

    pub fn advance(&mut self, inputs: NetworkInput) -> anyhow::Result<NetworkOutput> {
        unsafe {
            let out = twm::TWMNetworkRNNAdvanceNetwork(
                self.network_ptr,
                inputs.as_ptr() as *mut _,
                INPUT_SIZE,
            );

            if out.is_null() {
                anyhow::bail!("Advance failed");
            }

            let mut result = [0.0; OUTPUT_SIZE as usize];
            result.copy_from_slice(std::slice::from_raw_parts(out, OUTPUT_SIZE as usize));

            Ok(result)
        }
    }

    pub fn update(&mut self, expected_output: NetworkOutput) -> anyhow::Result<()> {
        unsafe {
            if twm::TWMNetworkRNNUpdateNetwork(self.network_ptr, expected_output.as_ptr() as *mut _)
                != 0
            {
                anyhow::bail!("Update failed");
            }
        }
        Ok(())
    }

    pub fn update_learning_rate(&mut self, rate: f64) {
        unsafe {
            twm::TWMNetworkRNNUpdateLearningRate(self.network_ptr, rate);
        }
    }

    pub fn reset(&mut self) {
        unsafe {
            twm::TWMNetworkRNNResetNetwork(self.network_ptr);
        }
    }

    pub fn as_ptr(&self) -> twm::PTWM_RNN_NETWORK {
        self.network_ptr
    }
}
