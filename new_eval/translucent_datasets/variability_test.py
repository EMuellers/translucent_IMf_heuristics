from new_eval.translucent_datasets.generate_translucent_dataset import variant_ratio, translucent_variant_ratio
import pandas as pd

log = pd.read_csv(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\road_traffic_fine\road_traffic_fine_0.2.csv")
print(f'Variant ratio of the log: {variant_ratio(log)}')
print(f'Translucent variant ratio of the log: {translucent_variant_ratio(log)}')
print()