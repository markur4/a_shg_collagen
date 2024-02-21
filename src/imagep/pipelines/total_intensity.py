"""Pipeline to calculate the total intensity of an image"""

# %%
from typing import Callable
import numpy as np
import pandas as pd

# > Local


# %%
class TotalIntensity:
    """Pipeline to calculate the total intensity of an image"""

    def __init__(self, imgs, **kws):
        """Initialize the TotalIntensity class
        :param imgs: Image data
        :type imgs: np.ndarray
        """
        self.imgs = imgs
        self.imgs = self.imgs if self.imgs.ndim == 3 else self.imgs[None, ...]
        self.imgs = self.imgs.astype(np.float64)

    def run(self):
        """Calculate the total intensity of the image"""
        self.intensity = np.sum(self.imgs)
        return self.intensity

    def __repr__(self):
        return f"TotalIntensity: {self.intensity}"


# %%
# == Import ============================================================

if __name__ == "__main__":
    parent = (
        "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/240210 Immunocyto2/"
    )
    paths = [
        # parent + "Exp. 1/LTMC Dmp1/",
        parent + "Exp. 1/LTMC Sost/",
        # parent + "Exp. 2/LTMC Dmp1/",
        parent + "Exp. 2/LTMC Sost/",
        # parent + "Exp. 2/2D Dmp1/",
        parent + "Exp. 2/2D Sost/",
        # parent + "Exp. 3/LTMC Dmp1/",
        parent + "Exp. 3/LTMC Sost/",
        # parent + "Exp. 3/2D Dmp1/",
        parent + "Exp. 3/2D Sost/",
    ]
    # > contains e.g.: "D0 LTMC DAPI 40x.tif"

    import imagep as ip
    from pathlib import Path

    # > IA_ori = original
    IA_ori_sost = ip.Stack(
        data=paths,
        fname_pattern="*Sost*.tif",
        sort=False,
        imgname_position=0,
    )
    print(IA_ori_sost.imgs.shape, IA_ori_sost.imgs.dtype)
    _img = IA_ori_sost.imgs[0]
    print(_img.shape, _img.dtype)
    # %%
    ### imshow
    IA_ori_sost.imshow(save_as="5_Sost", batch_size=4)  #:: uncomment

    # %%
    ### Sum up the intensity
    def _calc_sum(img: np.ndarray):
        """Calculate the sum of the intensity"""
        intensity = img.sum()
        return round(intensity.item(), 1)

    def calc_and_record(imgs: ip.l2Darrays, op: Callable):
        """Calculates percentage of area covered by nuclei"""

        data = []
        for img in imgs:
            ### Calculate total intensity
            result = op(img)

            ### Construct Record
            path = Path(img.folder).parts
            record = (*path, img.name, result)
            data.append(record)

        return data

    intens_sost = calc_and_record(IA_ori_sost.imgs, op=_calc_sum)
    intens_sost

    # %%
    ### For DMP1
    paths = [
        parent + "Exp. 1/LTMC Dmp1/",
        # parent + "Exp. 1/LTMC Sost/",
        parent + "Exp. 2/LTMC Dmp1/",
        # parent + "Exp. 2/LTMC Sost/",
        parent + "Exp. 2/2D Dmp1/",
        # parent + "Exp. 2/2D Sost/",
        parent + "Exp. 3/LTMC Dmp1/",
        # parent + "Exp. 3/LTMC Sost/",
        parent + "Exp. 3/2D Dmp1/",
        # parent + "Exp. 3/2D Sost/",
    ]
    # > contains e.g.: "D0 LTMC DAPI 40x.tif"

    import imagep as ip
    from pathlib import Path

    # > IA_ori = original
    IA_ori_dmp1 = ip.Stack(
        data=paths, fname_pattern="*Dmp1*.tif", sort=False, imgname_position=0
    )
    print(IA_ori_sost.imgs.shape, IA_ori_sost.imgs.dtype)
    _img = IA_ori_sost.imgs[0]
    print(_img.shape, _img.dtype)

    # %%
    ### imshow
    IA_ori_dmp1.imshow(save_as="6_Dmp1", batch_size=4)  #:: uncomment

    # %%
    ### Sum up the intensity
    intens_dmp1 = calc_and_record(IA_ori_dmp1.imgs, op=_calc_sum)
    intens_dmp1

    # %%
    def records_to_df(list_records: list[tuple]) -> pd.DataFrame:
        """Converts dictionary to dataframe. Dictionary has tuple of
        indices as keys and a scalar as value. Indices start with an
        integer row index. Returns a dataframe with indices expanded to
        columns.
        """

        ### Convert the list of records into a DataFrame
        df = pd.DataFrame(
            list_records,
            columns=[
                "folder1",
                "folder2",
                "img_name",
                "result",
            ],
        )
        return df

    intens = intens_sost + intens_dmp1
    df_intens = records_to_df(intens)
    df_intens
    # %%
    ### save dataframe as xlsx
    df_intens.to_excel("7_total_intensity.xlsx", index=False)
    # %%
    # =================================================================
    # == Merge with nuclei
    ### Import
    df_nuclei = pd.read_excel("4_perc_nuclei.xlsx")
    df_nuclei

    # %%
    def process_df(df: pd.DataFrame):
        """Processes the dataframe"""
        ### split column img_name and retrieve index 0
        df["day"] = df["img_name"].str.split("*").str[0]
        ## turn day into integer
        df["day"] = df["day"].str.replace("D", "").astype(int)

        # df["matrix"] = df["folder2"].str.split(" ").str[0]
        # df["gene"] = df["folder2"].str.split(" ").str[1]

        return df

    df_nuclei = process_df(df_nuclei)
    df_nuclei
    # %%
    df_intens = process_df(df_intens)
    df_intens

    # %%
    ### merge them together
    df = pd.merge(df_nuclei, df_intens, on=["folder1", "folder2", "day"])
    df
    # %%
    ### Make table pretty
    _df_p = (
        df.rename(
            columns={
                "result_x": "perc_nuclei",
                "result_y": "total_intensity",
                "folder1": "rep",
            },
        ).assign(
            **{
                "matrix": df["folder2"].str.split(" ").str[0],
                "gene": df["folder2"].str.split(" ").str[1],
                ### make day categorical
                "day": pd.Categorical(
                    df["day"],
                    categories=[0, 7, 14, 21],
                    ordered=True,
                ),
            },
        )
        ### Remove columns
        .drop(columns=["img_name_x", "img_name_y", "folder2"])
    )
    _df_p
    # %%
    # === Analysis =====================================================
    ### Turn logarithmic scale into linear by 2^x
    # _df_p["2^perc_nuclei"] = 2 ** _df_p["perc_nuclei"]

    # _df_p["-log10(perc_nuclei)"] = -np.log10( _df_p["perc_nuclei"])

    ### Calculate the ratio
    # _df_p["[RFU / Nuclei]"] = _df_p["total_intensity"] / _df_p["2^perc_nuclei"]

    # _df_p["[RFU / Nuclei]"] = _df_p["total_intensity"] / _df_p["-log10(perc_nuclei)"]

    _df_p["[RFU / Nuclei]"] = _df_p["total_intensity"] / _df_p["perc_nuclei"]
    _df_p
    # %%
    ### Save as xlsx
    _df_p.to_excel("8_data.xlsx", index=False)

    # %%
    _df_p.dtypes
    # %%
    ### Plot
    import seaborn as sns

    # sns.catplot(
    #     kind="strip",
    #     dodge=True,
    #     data=_df_p,
    #     sharey=False,
    #     **dims,
    # )

    # %%
    ### Plot with plotastic
    import matplotlib.pyplot as plt
    import plotastic as plst

    def plot(y, save_as):
        dims = dict(
            x="day",
            y=y,
            col="matrix",
            hue="gene",
        )

        DA = plst.DataAnalysis(data=_df_p, dims=dims, subject="rep")
        DA
        DA = (
            DA.subplots(sharey=True, wspace=0.1, figsize=(3.5, 2))
            .fillaxes(kind="strip", dodge=True, alpha=0.7, jitter=0.03)
            .edit_y_scale_log()
            .edit_legend()
            #   .plot_connect_subjects()
        )

        plt.savefig(save_as, dpi=300)

    plot(y="[RFU / Nuclei]", save_as="8_plot.png")
    # %%
    plot(y="total_intensity", save_as="8_plot_total_intensity.png")
    # %%
    plot(y="perc_nuclei", save_as="8_plot_perc_nuclei.png")
