import pandas as pd
from pycocotools.coco import COCO


def object_gender_ratio_in_coco(df_captions):
    # ANNOTATIONS
    coco = COCO(
        "/image/nlp-datasets/laura/data/mscoco/datasets/annotations2014/instances_val2014.json"
    )
    # VALIDATION DATA
    img_idx_minival = list(set(df_captions["id"]))
    print(f"{len(img_idx_minival)} images.")
    img_idx_minival = list(set(df_captions.loc[df_captions["gender"] != "Empty", "id"]))
    print(f"{len(img_idx_minival)} images with a person detected.")

    # Get list of image_ids which contain the object i (e.g. bicycles)
    objects = dict()
    objects_image_ids = dict()
    cat_ids = coco.getCatIds()

    map_gender = {"Unsure": 0, "Male": 1, "Female": 2}
    for cat_id in cat_ids:
        objects[cat_id] = [0, 0, 0]
        img_idx_val2014 = coco.getImgIds(catIds=[cat_id])
        objects_image_ids[cat_id] = list(
            filter(lambda x: x in img_idx_minival, img_idx_val2014)
        )
        # add +1 to this object wrt the main gender in the image
        for img_idx in objects_image_ids[cat_id]:
            gender = list(df_captions.loc[df_captions["id"] == img_idx, "gender"])[0]
            objects[cat_id][map_gender[gender]] += 1

    super_dict = []
    for obj_id, counter in objects.items():
        cats = coco.loadCats(obj_id)[0]
        if sum(counter) > 0:
            #     ratio = float(counter[1] / sum(counter))
            bias_male = float(counter[1] / sum(counter))
            bias_female = float(counter[2] / sum(counter))
            super_dict.append(
                {
                    "id": obj_id,
                    "supercategory": cats["supercategory"],
                    "object": cats["name"],
                    "Male": counter[1],
                    "Female": counter[2],
                    "Neutral": counter[0],
                    "bias_male": bias_male,
                    "bias_female": bias_female,
                }
            )
        else:
            # print(f"Object {cats['name']} was not in text.")
            pass
    counter_df = pd.DataFrame(super_dict)
    try:
        counter_df.sort_values("bias_male", ascending=False, inplace=True)
    except KeyError:
        print(super_dict)
    return counter_df


if __name__ == "__main__":
    import sys

    af = sys.argv[1]
    df = pd.read_csv(af, sep="\t")
    counter_df = object_gender_ratio_in_coco(df)
    print(counter_df)
