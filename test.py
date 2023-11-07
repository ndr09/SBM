import tests_cart_pole_nn
import tests_lunar_lander_nn
import tests_cart_pole_snn_nocode
import tests_cart_pole_snn_ratecode
import tests_cart_pole_snn_latencycode
import tests_cart_pole_snn_phasecode
import os

if __name__ == "__main__":
    os.makedirs("./results_granata", exist_ok=True)

    os.makedirs("./results_granata/cart_pole_NN", exist_ok=True)
    if not os.path.exists("./results_granata/cart_pole_NN/CartPole_NN.png"):
        tests_cart_pole_nn.cart_pole_NN("./results_granata/cart_pole_NN")

    os.makedirs("./results_granata/lunar_lander_NN", exist_ok=True)
    if not os.path.exists("./results_granata/lunar_lander_NN/lunar_lander_NN.png"):
        tests_lunar_lander_nn.lunar_lander_NN("./results_granata/lunar_lander_NN")

    os.makedirs("./results_granata/cart_pole_SNN_nocode", exist_ok=True)
    if not os.path.exists("./results_granata/cart_pole_SNN_nocode/cart_pole_SNN_nocode.png"):
        tests_cart_pole_snn_nocode.cart_pole_SNN_nocode("./results_granata/cart_pole_SNN_nocode")

    os.makedirs("./results_granata/cart_pole_SNN_ratecode", exist_ok=True)
    if not os.path.exists("./results_granata/cart_pole_SNN_ratecode/cart_pole_SNN_ratecode.png"):
        tests_cart_pole_snn_ratecode.cart_pole_SNN_ratecode("./results_granata/cart_pole_SNN_ratecode")

    os.makedirs("./results_granata/cart_pole_SNN_latencycode", exist_ok=True)
    if not os.path.exists("./results_granata/cart_pole_SNN_latencycode/cart_pole_SNN_latencycode.png"):
        tests_cart_pole_snn_latencycode.cart_pole_SNN_latencycode()

    os.makedirs("./results_granata/cart_pole_SNN_phasecode", exist_ok=True)
    if not os.path.exists("./results_granata/cart_pole_SNN_phasecode/cart_pole_SNN_phasecode.png"):
        tests_cart_pole_snn_phasecode.cart_pole_SNN_phasecode()