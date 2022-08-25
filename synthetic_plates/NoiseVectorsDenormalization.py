import pandas as pd
import numpy as np
import os

ranges = pd.read_csv("utils/noise/noises_parameters_ranges.csv")
ranges.set_index("Name", inplace=True)
address = "utils/noise/"
nvs = pd.read_csv(address + "noise_vectors(weighted).csv")


def transform_back(X, min, max):
    return (X * (max - min)) + min


for i in range(1, 11):
    nvs['image_' + str(i)] = nvs['image_' + str(i)].astype(np.float16)
    nvs['image_' + str(i)] = transform_back(nvs['image_' + str(i)], float(ranges.loc["image_min"]),
                                            float(ranges.loc["image_max"]))

nvs['LightNoise'] = nvs['LightNoise'].astype(np.float16)
nvs['LightNoise'] = transform_back(nvs['LightNoise'], float(ranges.loc["LightNoise_min"]),
                                   float(ranges.loc["LightNoise_max"]))
nvs['LightNoise_light_param'] = nvs['LightNoise_light_param'].astype(np.float16)
nvs['LightNoise_light_param'] = transform_back(nvs['LightNoise_light_param'],
                                               float(ranges.loc["LightNoise_light_param_min"]),
                                               float(ranges.loc["LightNoise_light_param_max"]))
nvs['GradientLightNoise'] = nvs['GradientLightNoise'].astype(np.float16)
nvs['GradientLightNoise'] = transform_back(nvs['GradientLightNoise'], float(ranges.loc["GradientLightNoise_min"]),
                                           float(ranges.loc["GradientLightNoise_max"]))
nvs['GradientLightNoise_light_param'] = nvs['GradientLightNoise_light_param'].astype(np.float16)
nvs['GradientLightNoise_light_param'] = transform_back(nvs['GradientLightNoise_light_param'],
                                                       float(ranges.loc["GradientLightNoise_light_param_min"]),
                                                       float(ranges.loc["GradientLightNoise_light_param_max"]))
nvs['CircularLightNoise'] = nvs['CircularLightNoise'].astype(np.float16)
nvs['CircularLightNoise'] = transform_back(nvs['CircularLightNoise'], float(ranges.loc["CircularLightNoise_min"]),
                                           float(ranges.loc["CircularLightNoise_max"]))
nvs['CircularLightNoise_light_param'] = nvs['CircularLightNoise_light_param'].astype(np.float16)
nvs['CircularLightNoise_light_param'] = transform_back(nvs['CircularLightNoise_light_param'],
                                                       float(ranges.loc["CircularLightNoise_light_param_min"]),
                                                       float(ranges.loc["CircularLightNoise_light_param_max"]))
nvs['CircularLightNoise_r_circle'] = nvs['CircularLightNoise_r_circle'].astype(np.float16)
nvs['CircularLightNoise_r_circle'] = transform_back(nvs['CircularLightNoise_r_circle'],
                                                    float(ranges.loc["CircularLightNoise_r_circle_min"]),
                                                    float(ranges.loc["CircularLightNoise_r_circle_max"]))
nvs['CircularLightNoise_kernel_sigma'] = nvs['CircularLightNoise_kernel_sigma'].astype(np.float16)
nvs['CircularLightNoise_kernel_sigma'] = transform_back(nvs['CircularLightNoise_kernel_sigma'],
                                                        float(ranges.loc["CircularLightNoise_kernel_sigma_min"]),
                                                        float(ranges.loc["CircularLightNoise_kernel_sigma_max"]))
nvs['SPNoise'] = nvs['SPNoise'].astype(np.float16)
nvs['SPNoise'] = transform_back(nvs['SPNoise'], float(ranges.loc["SPNoise_min"]), float(ranges.loc["SPNoise_max"]))
nvs['SPNoise_amount_sp'] = nvs['SPNoise_amount_sp'].astype(np.float16)
nvs['SPNoise_amount_sp'] = transform_back(nvs['SPNoise_amount_sp'], float(ranges.loc["SPNoise_amount_sp_min"]),
                                          float(ranges.loc["SPNoise_amount_sp_max"]))
nvs['SPNoise_bw'] = nvs['SPNoise_bw'].astype(np.float16)
nvs['SPNoise_bw'] = transform_back(nvs['SPNoise_bw'], float(ranges.loc["SPNoise_bw_min"]),
                                   float(ranges.loc["SPNoise_bw_max"]))
nvs['SPNoise_pepper_color'] = nvs['SPNoise_pepper_color'].astype(np.float16)
nvs['SPNoise_pepper_color'] = transform_back(nvs['SPNoise_pepper_color'], float(ranges.loc["SPNoise_pepper_color_min"]),
                                             float(ranges.loc["SPNoise_pepper_color_max"]))
nvs['SPNoise_salt_color'] = nvs['SPNoise_salt_color'].astype(np.float16)
nvs['SPNoise_salt_color'] = transform_back(nvs['SPNoise_salt_color'], float(ranges.loc["SPNoise_salt_color_min"]),
                                           float(ranges.loc["SPNoise_salt_color_max"]))
nvs['NegativeNoise'] = nvs['NegativeNoise'].astype(np.float16)
nvs['NegativeNoise'] = transform_back(nvs['NegativeNoise'], float(ranges.loc["NegativeNoise_min"]),
                                      float(ranges.loc["NegativeNoise_max"]))
nvs['BlurNoise'] = nvs['BlurNoise'].astype(np.float16)
nvs['BlurNoise'] = transform_back(nvs['BlurNoise'], float(ranges.loc["BlurNoise_min"]),
                                  float(ranges.loc["BlurNoise_max"]))
nvs['BlurNoise_blur_sigma'] = nvs['BlurNoise_blur_sigma'].astype(np.float16)
nvs['BlurNoise_blur_sigma'] = transform_back(nvs['BlurNoise_blur_sigma'], float(ranges.loc["BlurNoise_blur_sigma_min"]),
                                             float(ranges.loc["BlurNoise_blur_sigma_max"]))
nvs['BlurNoise_blur_kernel_size'] = nvs['BlurNoise_blur_kernel_size'].astype(np.float16)
nvs['BlurNoise_blur_kernel_size'] = transform_back(nvs['BlurNoise_blur_kernel_size'],
                                                   float(ranges.loc["BlurNoise_blur_kernel_size_min"]),
                                                   float(ranges.loc["BlurNoise_blur_kernel_size_max"]))
nvs['pitch'] = nvs['pitch'].astype(np.float16)
nvs['pitch'] = transform_back(nvs['pitch'], float(ranges.loc["pitch_min"]), float(ranges.loc["pitch_max"]))
nvs['yaw'] = nvs['yaw'].astype(np.float16)
nvs['yaw'] = transform_back(nvs['yaw'], float(ranges.loc["yaw_min"]), float(ranges.loc["yaw_max"]))
nvs['roll'] = nvs['roll'].astype(np.float16)
nvs['roll'] = transform_back(nvs['roll'], float(ranges.loc["roll_min"]), float(ranges.loc["roll_max"]))
nvs['scale'] = nvs['scale'].astype(np.float16)
nvs['scale'] = transform_back(nvs['scale'], float(ranges.loc["scale_min"]), float(ranges.loc["scale_max"]))

nvs.to_csv(address + "denormalized_noise_vectors(weighted).csv", index=False)
