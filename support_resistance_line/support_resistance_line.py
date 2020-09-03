import math
from collections import Iterable
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lazy_object_proxy.utils import cached_property
from sklearn import metrics
from sklearn.cluster import KMeans


class StraightLine:
    def __init__(
        self,
        x1: float = None,
        y1: float = None,
        x2: float = None,
        y2: float = None,
        slope: Optional[float] = None,
    ):

        if slope is not None:
            self.slope = slope
        else:
            if x1 == x2:
                self.slope = np.nan
            else:
                self.slope = (y2 - y1) / (x2 - x1)  # type: ignore
        self.intercept = y1 - self.slope * x1  # type: ignore

    def get_point_distance(self, x0: float, y0: float) -> float:
        return abs(self.slope * x0 - y0 + self.intercept) / math.sqrt(
            self.slope ** 2 + 1
        )

    def is_point_above_line(self, x0: float, y0: float) -> bool:
        pred_y = x0 * self.slope + self.intercept
        if pred_y == y0:
            print(f'Point ({x0}, {y0}) is on line y = {self.slope}x + {self.intercept}')
        return y0 > pred_y

    def predict(
        self, x_list: Iterable, limit: Optional[Iterable] = None
    ) -> List[float]:
        if not isinstance(x_list, Iterable):
            x_list = [x_list]
        results = [self.slope * _ + self.intercept for _ in x_list]
        if len(results) == 1:
            return results[0]
        if limit is not None:
            results = [
                _ if _ > min(limit) and _ < max(limit) else np.nan for _ in results
            ]
        return results


def clustering_kmeans(num_list: List[float], thresh: float = 0.03) -> List[float]:

    # support/resistance pos cluster starting from 1 to N
    k_rng = range(1, len(num_list) + 1)
    est_arr = [KMeans(n_clusters=k).fit([[num] for num in num_list]) for k in k_rng]

    # distance-sum of all cluster
    sum_squares = [e.inertia_ for e in est_arr]

    # ratio of distance-sum to which of only one cluster
    diff_squares = [squares / sum_squares[0] for squares in sum_squares]
    diff_squares_pd = pd.Series(diff_squares)

    # select cluster based on thresh
    thresh_pd = diff_squares_pd[diff_squares_pd < thresh]

    if len(thresh_pd) > 0:
        select_k = thresh_pd.index[0] + 1
    else:
        # if no such, select the most one
        select_k = k_rng[-1]

    est = est_arr[select_k - 1]
    results = est.predict([[num] for num in num_list])

    return results


class SupportResistanceLine:
    def __init__(
        self, data: pd.Series, kind: Literal['support', 'resistance'] = 'support'
    ):
        if not isinstance(data, pd.Series):
            raise TypeError('data should be pd.Series')

        self.y = data.reset_index(drop=True).rename('y').rename_axis('x')
        self.x = self.y.index.to_series()

        self.length = len(self.y)

        self.kind = kind
        self.dot_color = 'g' if kind == 'support' else 'r'

    @cached_property
    def twin(self):
        srl = SupportResistanceLine(
            self.y, 'resistance' if self.kind == 'support' else 'support'
        )
        srl.extreme_pos = self.extreme_pos  # avoid repeated calc
        return srl

    @cached_property
    def iterated_poly_fits(
        self,
    ) -> Tuple[pd.DataFrame, np.polynomial.chebyshev.Chebyshev]:
        fit_df = self.y.to_frame()
        rolling_window = int(len(self.y) / 30)
        fit_df['y_roll_mean'] = (
            fit_df['y'].rolling(rolling_window, min_periods=1).mean()
        )

        # the orginal mean distance of y and moving avg
        distance_mean = np.sqrt(
            metrics.mean_squared_error(fit_df.y, fit_df.y_roll_mean)
        )

        degree = int(len(self.y) / 40)
        poly = None
        y_fit = None
        while degree < 100:
            # iterate 1-100 degress of ploy
            poly = np.polynomial.Chebyshev.fit(self.x, self.y, degree)
            y_fit = poly(self.x)
            fit_df[f'poly_{degree}'] = y_fit
            # mean distance of y and y_fit
            distance_fit = np.sqrt(metrics.mean_squared_error(fit_df.y, y_fit))
            if distance_fit <= distance_mean * 0.6:
                # stop iteration when distance_fit <= distance_mean * 0.6, indicating a perfect trend line
                break
            degree += 1

        return fit_df, poly

    @cached_property
    def best_poly(self) -> np.polynomial.chebyshev.Chebyshev:
        return self.iterated_poly_fits[1]

    @cached_property
    def poly_degree(self) -> int:
        '''Degree(s) of the fitting polynomials'''
        return self.best_poly.degree()

    @cached_property
    def poly_fit(self) -> pd.Series:
        '''Fitted series'''
        return self.best_poly(self.x)

    def plot_poly(self, show=False):
        fig, ax = plt.subplots(1, figsize=(16, 9))
        df = self.iterated_poly_fits[0].assign(y=self.y, best_poly=self.poly_fit)
        df.plot(ax=ax, figsize=(16, 9), colormap='coolwarm')
        if show:
            plt.show()
        return fig, ax

    @cached_property
    def extreme_pos(self) -> Tuple[List[int], List[int]]:
        # roots derived function
        extreme_pos = [int(round(_.real)) for _ in self.best_poly.deriv().roots()]
        extreme_pos = [_ for _ in extreme_pos if _ > 0 and _ < self.length]

        # distinguish maximum and minimum using second derivative
        second_deriv = self.best_poly.deriv(2)
        min_extreme_pos = []
        max_extreme_pos = []
        for pos in extreme_pos:
            if second_deriv(pos) > 0:
                min_extreme_pos.append(pos)
            elif second_deriv(pos) < 0:
                max_extreme_pos.append(pos)

        return max_extreme_pos, min_extreme_pos

    def plot_extreme_pos(self, show: bool = False):
        max_extreme_pos, min_extreme_pos = self.extreme_pos

        fig, ax = plt.subplots(1, figsize=(16, 9))
        self.y.to_frame().assign(best_poly=self.poly_fit).plot(ax=ax)
        ax.scatter(
            min_extreme_pos, [self.best_poly(_) for _ in min_extreme_pos], s=50, c='g'
        )
        ax.scatter(
            max_extreme_pos, [self.best_poly(_) for _ in max_extreme_pos], s=50, c='r'
        )
        if show:
            plt.show()

        return fig, ax

    @cached_property
    def support_resistance_pos(self) -> List[int]:
        '''Real local extreme pos around roots'''

        def find_left_and_right_pos(pos, refer_pos):
            '''Find two resistance points around a support point, or vice versa'''
            refer_sr = pd.Series(refer_pos)
            left_pos = (
                refer_sr[refer_sr < pos].iloc[-1]
                if len(refer_sr[refer_sr < pos]) > 0
                else 0
            )
            right_pos = (
                refer_sr[refer_sr > pos].iloc[0]
                if len(refer_sr[refer_sr > pos]) > 0
                else self.length
            )
            return left_pos, right_pos

        def extreme_around(left_pos, right_pos):
            '''Locate real local extreme pos around roots'''

            if self.kind == 'support':
                extreme_around_pos = self.y.iloc[left_pos:right_pos].idxmin()
            else:  # resistance
                extreme_around_pos = self.y.iloc[left_pos:right_pos].idxmax()

            # If the extreme point is on the edge, meaning false point, discard
            if extreme_around_pos in (left_pos, right_pos):
                return 0

            return extreme_around_pos

        if self.kind == 'support':
            refer_pos, extreme_pos = self.extreme_pos
        else:
            extreme_pos, refer_pos = self.extreme_pos

        support_resistance_pos = []
        for _, pos in enumerate(extreme_pos):
            if pos in [0, self.length]:
                continue

            left_pos, right_pos = find_left_and_right_pos(pos, refer_pos)

            support_resistance_pos.append(extreme_around(left_pos, right_pos))

        # Deduplicate
        support_resistance_pos = sorted(set(support_resistance_pos))

        # Remove 0
        if 0 in support_resistance_pos:
            support_resistance_pos.remove(0)

        return support_resistance_pos

    @cached_property
    def support_resistance_df(self) -> pd.DataFrame:
        return (
            pd.Series(
                self.y.loc[self.support_resistance_pos],
                index=self.support_resistance_pos,
            )
            .sort_index()
            .rename_axis('x')
            .reset_index()
        )

    def plot_real_extreme_points(self, show: bool = False):
        return self.show_line(self.support_resistance_df, show=show)

    @cached_property
    def clustered_pos(self) -> List[int]:
        def clustering_nearest(num_list, thresh=self.length / 80):
            sr = pd.Series(num_list).sort_values().reset_index(drop=True)
            while sr.diff().min() < thresh:
                index1 = sr.diff().idxmin()
                index2 = index1 - 1
                num1 = sr[index1]
                num2 = sr[index2]
                y1 = self.y.iloc[num1]
                y2 = self.y.iloc[num2]

                smaller_y_index = index1 if y1 < y2 else index2
                bigger_y_index = index1 if y1 > y2 else index2
                sr = sr.drop(
                    bigger_y_index if self.kind == 'support' else smaller_y_index
                ).reset_index(drop=True)
            return sr.tolist()

        clustered_pos = clustering_nearest(self.support_resistance_df['x'].tolist())
        return clustered_pos

    def plot_clustered_pos(self, show: bool = False):
        support_resistance_df = self.support_resistance_df.loc[
            lambda _: _['x'].isin(self.clustered_pos)
        ].copy()

        return self.show_line(support_resistance_df, show=show)

    def score_lines_from_a_point(
        self, last_support_resistance_pos: pd.Series
    ) -> pd.DataFrame:
        '''Assign scores to all lines through a point'''

        # Only include points before the point
        support_resistance_df = self.support_resistance_df.loc[
            lambda _: _['x'] <= last_support_resistance_pos['x']
        ].copy()

        if len(support_resistance_df) <= 2:
            return pd.DataFrame()

        # Calc slopes of lines through each points
        support_resistance_df['slope'] = support_resistance_df.apply(
            lambda _: StraightLine(
                _['x'],
                _['y'],
                last_support_resistance_pos['x'],
                last_support_resistance_pos['y'],
            ).slope,
            axis=1,
        )

        # Rank lines based on slope
        if self.kind == 'support':
            support_resistance_df = support_resistance_df.dropna().sort_values('slope')
        elif self.kind == 'resistance':
            support_resistance_df = support_resistance_df.dropna().sort_values(
                'slope', ascending=False
            )

        # Filter out lines that are too cliffy
        support_resistance_df = support_resistance_df[
            support_resistance_df['slope'].abs() / self.y.mean() < 0.003
        ]
        if len(support_resistance_df) <= 2:
            return pd.DataFrame()

        # Cluster
        thresh = 0.03
        support_resistance_df['cluster'] = clustering_kmeans(
            support_resistance_df['slope'], thresh
        )
        while (
            support_resistance_df.groupby('cluster').apply(len).max() <= 2
        ):  # If num of points in the cluster with most point is still less than 2
            thresh *= 2
            if thresh >= 1:
                return pd.DataFrame()
            support_resistance_df['cluster'] = clustering_kmeans(
                support_resistance_df['slope'], thresh
            )

        def calc_score_for_cluster(cluster_df):
            if len(cluster_df) <= 2:
                return pd.DataFrame()

            avg_x = cluster_df.iloc[:-1]['x'].mean()
            avg_y = cluster_df.iloc[:-1]['y'].mean()
            line = StraightLine(
                cluster_df.iloc[-1]['x'],
                cluster_df.iloc[-1]['y'],
                slope=cluster_df.iloc[-1]['slope'],
            )
            mean_distance = line.get_point_distance(avg_x, avg_y)
            std = cluster_df.iloc[:-1]['x'].std(ddof=0)
            mean_x = cluster_df.iloc[:-1]['x'].mean()

            return pd.DataFrame(
                {
                    'cluster': cluster_df.name,
                    'x1': last_support_resistance_pos['x'],
                    'y1': last_support_resistance_pos['y'],
                    'x2': cluster_df.iloc[-1]['x'],
                    'y2': cluster_df.iloc[-1]['y'],
                    'slope': cluster_df.iloc[-1]['slope'],
                    'count': len(cluster_df) - 1,
                    'mean_distance': mean_distance,
                    'mean_x': mean_x,
                    'std': std,
                },
                index=[0],
            )

        score_df = (
            support_resistance_df.groupby('cluster')
            .apply(calc_score_for_cluster)
            .reset_index(drop=True)
        )

        # And the full points without clustering also should be included
        all_df = support_resistance_df.copy()
        all_df.name = 'all'
        score_df.loc[len(score_df)] = calc_score_for_cluster(all_df).iloc[0]

        return score_df

    def show_line(
        self,
        points_df: pd.DataFrame,
        *straight_line_list: StraightLine,
        show: bool = False,
    ):
        fig, ax = plt.subplots(1, figsize=(16, 9))
        self.y.to_frame().assign(best_poly=self.poly_fit).plot(ax=ax)

        # Green support dots, red resistance dots
        ax.scatter(
            points_df.x, points_df.y, s=50, c=self.dot_color, label=f'{self.kind}_dots'
        )

        for i, st_line in enumerate(straight_line_list):
            ax.plot(
                self.x,
                st_line.predict(self.x, limit=(self.y.min(), self.y.max())),
                label=(['1st', '2nd', '3rd'] + list('456789abcdefghijklmnopq'))[i],
            )
        plt.legend()
        if show:
            plt.show()

        return fig, ax

    @cached_property
    def last_area_support_resistance_df(self) -> pd.DataFrame:
        '''Find best lines for the 40% right-most points'''
        last_area_support_resistance_df = self.support_resistance_df[
            self.support_resistance_df['x'] > self.length * 0.75
        ].copy()

        df_list = [
            self.score_lines_from_a_point(row)
            for index, row in last_area_support_resistance_df.iterrows()
        ]

        last_area_support_resistance_df = pd.concat(df_list)

        if len(last_area_support_resistance_df) == 0:
            raise ValueError(
                f"Faild finding {self.kind} line, may due to a too short time series"
            )

        last_area_support_resistance_df['score'] = (
            last_area_support_resistance_df['mean_distance']
            / last_area_support_resistance_df['mean_x']
            / last_area_support_resistance_df['std']
        ).rank() / last_area_support_resistance_df['count']

        last_area_support_resistance_df = last_area_support_resistance_df.sort_values(
            ['score', 'count'], ascending=[True, False]
        ).reset_index(drop=True)

        return last_area_support_resistance_df

    def plot_top_lines(self, num=3, show=False):
        '''Plot the best 3 lines'''
        return self.show_line(
            self.support_resistance_df,  # 描点
            *(
                self.last_area_support_resistance_df[:num]
                .apply(
                    lambda _: StraightLine(_['x1'], _['y1'], _['x2'], _['y2']), axis=1
                )
                .tolist()
            ),
            show=show,
        )

    @cached_property
    def best_line(self) -> StraightLine:
        best_line_data = self.last_area_support_resistance_df.iloc[0]
        best_line = StraightLine(
            best_line_data['x1'],
            best_line_data['y1'],
            best_line_data['x2'],
            best_line_data['y2'],
        )
        return best_line

    def plot_best_line(self, show: bool = False):
        return self.show_line(self.support_resistance_df, self.best_line, show=show)

    def plot_steps(self):
        if self.kind != 'support':
            raise ValueError("Only support line object can call this method")
        print('Looking for best polynominal curve...')
        self.plot_poly(show=True)

        print('Looking for extreme pos of fitted curve...')
        self.plot_extreme_pos(show=True)

        print('Looking for support pos...')
        self.plot_real_extreme_points(show=True)

        print('Clusterinig support pos...')
        self.plot_clustered_pos(show=True)

        print('Iterate over support lines starting from the right most area...')
        self.last_area_support_resistance_df

        resistance_line = self.twin

        print('Looking for resistance pos...')
        resistance_line.plot_real_extreme_points(show=True)

        print('Clustring resistance pos...')
        resistance_line.plot_clustered_pos(show=True)

        print('Iterate over resistance lines starting from the right most area...')
        resistance_line.last_area_support_resistance_df

        self.resistance_line = resistance_line

        print('Plotting...')
        self.plot_both()

    def plot_both(self, ax=None, show: bool = False):
        if self.kind != 'support':
            raise ValueError("Only support line object can call this method")

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(16, 9))
        else:
            fig = ax.get_figure()

        self.y.to_frame().assign(best_poly=self.poly_fit).plot(ax=ax)

        ax.scatter(
            self.support_resistance_df.x,
            self.support_resistance_df.y,
            s=50,
            c=self.dot_color,
            label='support_dots',
        )

        resistance_line = self.twin

        ax.scatter(
            resistance_line.support_resistance_df.x,
            resistance_line.support_resistance_df.y,
            s=50,
            c=resistance_line.dot_color,
            label='resistance_dots',
        )

        ax.plot(
            self.x,
            self.best_line.predict(self.x, (self.y.min(), self.y.max())),
            label='support_line',
            c='g',
        )
        ax.plot(
            self.x,
            resistance_line.best_line.predict(self.x, (self.y.min(), self.y.max())),
            label='resistance_line',
            c='r',
        )

        ax.legend()

        if show:
            plt.show()

        return fig, ax
