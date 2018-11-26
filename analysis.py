import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from astropy.stats import circcorrcoef
import astropy.units as u

class gridCells:

    def __init__(self, dir, type):
        assert dir is not None
        
        self.dir = dir
        self.type = type
        self.move_thresh = 0.01

    def mean_phase_map(self, arr, bin_size):
        mpm_dict = {}
        for ybin in range(0, int(arr[:, 1].max() + 1), bin_size):
            mpm_dict[ybin] = {}
            for xbin in range(0, int(arr[:, 0].max() + 1), bin_size):
                phases = []
                mpm_dict[ybin][xbin] = []
                for spike in arr:
                    if (xbin <= spike[0] <= xbin + bin_size) and (ybin <= spike[1] <= ybin + bin_size):
                        phases.append(spike[2])
                mpm_dict[ybin][xbin] = np.nanmean(np.asarray(phases))
        return mpm_dict

    def adjacent_matrix(self, cell, phase):
        """Determines change vector from central cell to cell
        nearest in value in 5x5 IN FORM **[X,Y]** """        
        x = int(cell[0])
        y = int(cell[1])
        y_size = self.arena_size[0]-1
        a = self.padded_phase_df.iloc[y_size-y:y_size-y+5,x:x+5]

        try:
            nearest = np.nanargmin(np.abs(a-phase))
            loc = [(nearest%5)-2,2-(nearest//5)]

            #Rounding down to the nearest bin, adding 0.5 to point to center of bin        
            xp = x+loc[0]+0.5
            yp = y+loc[1]+0.5    
            return[xp - cell[0],yp - cell[1]]

        except:
            return [0,0]    


    def adjacent_spikes(self, spikes, phase):
        """Get location of spike with most similar phase"""
        y_size = self.arena_size[0] - 1
        phases = []
        for i in spikes:
            x = int(i[0])
            y = int(i[1])
            phases.append(self.phase_df.iloc[y_size - y, x])
        phases = np.asarray(phases)
        try:
            nearest = np.nanargmin(np.abs(phases - phase))
        except:
            nearest = 0

        try:
            #Rounding down to the nearest bin, adding 0.5 to point to center of bin
            x = int(spikes[nearest][0])+0.5
            y = int(spikes[nearest][1])+0.5
            return [x - spikes[0, 0], y - spikes[0, 1]]

        except:
            return [0.0, 0.0]

    @staticmethod
    def vector_angles(df):
        """Basic formula for determining angle between vectors"""

        angles = []
        for index, row in df.iterrows():
            p0 = [row['Xdif'], row['Ydif']]
            p1 = [0, 0]
            p2 = [row['Xdif Predicted'], row['Ydif Predicted']]
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            atan = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            angles.append(np.degrees(atan))

        return angles

    @staticmethod
    def abs_vector_angles(arr):
        """Determines the angle between the horizontal axis (+1 x, +0 y)
        and the current vector, returns [observed,predicted]"""

        obs_angles = []
        pred_angles = []

        for i in range(len(arr) - 1):
            p0 = [arr[i, 1] + 1, arr[i, 2]]
            p1 = [arr[i, 1], arr[i, 2]]
            p2 = [arr[i + 1, 1], arr[i + 1, 2]]
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            atan = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            if np.degrees(atan) < 0:
                obs_angles.append(360 + np.degrees(atan))
            else:
                obs_angles.append(np.degrees(atan))

        for i in range(len(arr) - 1):
            p0 = [arr[i, 1] + 1, arr[i, 2]]
            p1 = [arr[i, 1], arr[i, 2]]
            p2 = [arr[i, 1] + arr[i, 7], arr[i, 2] + arr[i, 8]]
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            atan = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            if np.degrees(atan) < 0:
                pred_angles.append(360 + np.degrees(atan))
            else:
                pred_angles.append(np.degrees(atan))

        return (list(zip(obs_angles, pred_angles)))

    def phase_analysis(self):
        """Loads and normalizes data, generates predictions based on
        mean phase map and saves everything in a dataframe"""

        self.XYspkT = np.loadtxt(self.dir + 'XYspkT.csv', delimiter=',')

        # Normalizes position data to [0,0] alignment in bottom-left corner
        self.XYspkT[:,1] -= self.XYspkT[:,1].min()
        self.XYspkT[:,0] -= self.XYspkT[:,0].min()
        self.scaled_XY = self.XYspkT/2

        # Load precise trajectory
        self.xyPos = np.loadtxt(self.dir + 'xyPos.csv', delimiter=',')[::25]
        self.xyPos[:,1] -= self.xyPos[:,1].min()
        self.xyPos[:,0] -= self.xyPos[:,0].min()

        # Load spike times
        self.spkT = np.loadtxt(self.dir + 'spkT.csv', delimiter=',')

        # Load phase data
        self.phase = np.loadtxt(self.dir + 'Phase.csv', delimiter=',')
        self.scaled_phase = self.phase - 3.14

        # Generate mean phase map
        mpm_arr = np.column_stack((self.XYspkT, self.scaled_phase))
        mpm_dict = self.mean_phase_map(mpm_arr, 2)
        # Rotate the dataframe 90 CCW
        mpm = pd.DataFrame.from_dict(mpm_dict).T
        self.phase_df = mpm.reindex(index=mpm.index[::-1])
        self.arena_size = self.phase_df.shape

        # A padded phase map for extracting the 5x5 adjacency matrix (needed for edge values)
        padded_phase_map = np.pad(self.phase_df, pad_width=2, mode='constant', constant_values=np.nan)
        self.padded_phase_df = pd.DataFrame(data=padded_phase_map)

        # Main analysis
        # 1) Combine all data and sort by spike times
        unsorted = np.column_stack((self.spkT, self.scaled_XY, self.scaled_phase))
        sorted = unsorted[unsorted[:, 0].argsort()]

        # 2) Calculate movement magnitudes
        xdif = np.append(sorted[1:, 1], 0) - np.append(sorted[:-1, 1], 0)
        ydif = np.append(sorted[1:, 2], 0) - np.append(sorted[:-1, 2], 0)

        # 3) Drop rows with movements below threshold
        raw = np.column_stack((sorted, xdif, ydif))
        movement = raw[np.any(abs(raw[:, 4:]) >= self.move_thresh, axis=1)]

        # 4) Recalculate movement magnitudes as in 2)
        xdif = np.append(movement[1:, 1], 0) - np.append(movement[:-1, 1], 0)
        ydif = np.append(movement[1:, 2], 0) - np.append(movement[:-1, 2], 0)

        # 5) Combine all data, load next spikes phases in row for convenience
        next_phase = np.insert(movement[1:, 3], -1, 0)
        combined = np.column_stack((movement[:, :4], next_phase, xdif, ydif))

        # 6) Generate predictions

        #Spatial Analysis
        if self.type == 'spatial':
            predicted = [self.adjacent_matrix([i[1], i[2]], i[4]) for i in combined]
            predicted_movement = np.asarray(predicted)

        # Temporal Analysis
        elif self.type == 'temporal':
             predicted = [self.adjacent_spikes(combined[i:i + 6, 1:3], combined[i, 4]) for i in range(len(combined))]
             predicted_movement = np.asarray(predicted)

        # 7) Load all data into dataframe
        self.all = np.column_stack((combined, predicted_movement))
        self.df = pd.DataFrame(data=self.all, columns=['Time', 'X', 'Y', 'Phase', 'Next Phase', 'Xdif', 'Ydif', 'Xdif Predicted', 'Ydif Predicted'])

        self.angles = np.asarray(self.abs_vector_angles(self.all))
        
        # Generate observed/predicted circular correlation coefficient
        # Specify astropy degree units
        #alpha = self.angles[:,0]*u.deg
        #beta = self.angles[:,1]*u.deg
        #r = circcorrcoef(alpha, beta)
        
        r,p = pearsonr(self.angles[:,0],self.angles[:,1])
        
        return r

# class figureGenerator:
#
#     def __init__(self):
#         pass

    def XY_plot(self):
        """Plot of scaled spike positions"""

        plt.plot(self.scaled_XY[:, 0], self.scaled_XY[:, 1], '.')
        plt.show()

    def phase_plot(self):
        """Generates a radial distribution of all spike phases"""
        phase_degrees = np.degrees(self.phase)
        sorted_phase = np.sort(phase_degrees)

        # Radial histogram of phase distribution
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        theta = np.radians(np.arange(0, 366, 6))

        inds = [np.where(sorted_phase < i)[0][-1] for i in np.arange(6, 366, 6)]  # vectorize operation?
        b = np.split(sorted_phase, inds)
        radii = np.array([i.size / 3.2 for i in b])

        width = np.radians(360 / 60)
        bars = ax.bar(theta, radii, width=width, bottom=0.0)
        for theta, bar in zip(theta, bars):
            bar.set_facecolor(plt.cm.hsv(theta / 6.28))
            bar.set_alpha(0.5)

        plt.show()

    def phase_map_plot(self):
        """Generates a heatmap of the iso phase map"""
        plt.matshow(self.phase_df)

        # To-do: reflect radians in bar
        plt.colorbar()
        plt.show()

    def trajectory_plot(self):
        """Generates a plot of the complete trajectory data"""
        plt.plot(self.xyPos[:,0], self.xyPos[:,1], color='b')
        plt.plot(self.XYspkT[:,0], self.XYspkT[:,1], '.', color='r', markersize=6)
        plt.show()

    def prediction_plot(self):
        """Generates a plot with trajectory, spike positions, and prediction vectors"""
        plt.rcParams['figure.figsize'] = [20, 15]
        ax = plt.subplot()
        plt.plot(self.xyPos[:,0]/2, self.xyPos[:,1]/2, color='b')
        plt.plot(self.scaled_XY[:,0], self.scaled_XY[:,1], '.', color='r', markersize=6)
        for index, row in self.df.iterrows():
            # if (row['Xdif Predicted']>0) or (row['Ydif Predicted']>0):
            plt.arrow(row['X'], row['Y'], row['Xdif Predicted'], row['Ydif Predicted'],
                        head_width=0.2, color='black')

        ax.set_xticks(np.arange(self.scaled_XY[:,0].min()-1,self.scaled_XY[:,0].max()+2))
        ax.set_yticks(np.arange(self.scaled_XY[:,1].min()-1,self.scaled_XY[:,1].max()+1))
        ax.grid()
        plt.show()

    def corr_plot(self):
        """Generates simple bivariate distribution for correlation"""
        corr_df = pd.DataFrame(data=self.angles, columns=['Observed Heading Direction', 'Predicted Heading Direction'])
        sns.jointplot(x='Observed Heading Direction', y='Predicted Heading Direction', data=corr_df, kind='kde')
        plt.ylim(0, None)
        plt.xlim(0.1, None)
        plt.show()

    def corr_hex(self):
        """Generates simple bivariate distribution for correlation (hex form)"""
        corr_df = pd.DataFrame(data=self.angles, columns=['Observed Heading Direction', 'Predicted Heading Direction'])
        sns.jointplot(x='Observed Heading Direction', y='Predicted Heading Direction', data=corr_df, kind='hex')
        plt.show()

    def corr_heatmap(self):
        """Generates correlation heatmap"""
        heatmap, xedges, yedges = np.histogram2d(self.angles[:, 0], self.angles[:, 1], bins=40)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='afmhot')
        plt.colorbar()
        plt.ylabel('Predicted Heading Direction')
        plt.xlabel('Observed Heading Direction')
        plt.show()

# Tests:
# a = gridCells('datasets/1/',0.5,'temporal')
# a.phase_analysis()
# a.prediction_plot()


