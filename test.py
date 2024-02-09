import tests_cart_pole_nn
import tests_cart_pole_snn_nocode
import tests_cart_pole_snn_ratecode
import tests_cart_pole_snn_latencycode
import tests_cart_pole_snn_phasecode

if __name__ == "__main__":
    tests_cart_pole_nn.cart_pole_NN()
    tests_cart_pole_snn_nocode.cart_pole_SNN_nocode()
    tests_cart_pole_snn_ratecode.cart_pole_SNN_ratecode()
    tests_cart_pole_snn_latencycode.cart_pole_SNN_latencycode()
    tests_cart_pole_snn_phasecode.cart_pole_SNN_phasecode()