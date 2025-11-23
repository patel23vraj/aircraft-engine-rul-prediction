
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

extern "C" {
    // Fast rolling statistics computation
    void compute_rolling_stats(double* data, int size, int window, 
                              double* means, double* stds) {
        for (int i = 0; i < size; i++) {
            int start = std::max(0, i - window + 1);
            int count = i - start + 1;
            
            // Compute mean
            double sum = 0.0;
            for (int j = start; j <= i; j++) {
                sum += data[j];
            }
            means[i] = sum / count;
            
            // Compute std
            double var_sum = 0.0;
            for (int j = start; j <= i; j++) {
                double diff = data[j] - means[i];
                var_sum += diff * diff;
            }
            stds[i] = std::sqrt(var_sum / count);
        }
    }
    
    // Fast degradation pattern detection
    double detect_degradation_rate(double* sensor_data, int size) {
        if (size < 2) return 0.0;
        
        // Linear regression for trend
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        
        for (int i = 0; i < size; i++) {
            sum_x += i;
            sum_y += sensor_data[i];
            sum_xy += i * sensor_data[i];
            sum_x2 += i * i;
        }
        
        double slope = (size * sum_xy - sum_x * sum_y) / 
                      (size * sum_x2 - sum_x * sum_x);
        
        return slope;
    }
    
    // Fast feature correlation matrix
    void compute_correlation_matrix(double* features, int rows, int cols, 
                                   double* corr_matrix) {
        // Compute correlation coefficients
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == j) {
                    corr_matrix[i * cols + j] = 1.0;
                    continue;
                }
                
                double sum_x = 0, sum_y = 0, sum_xy = 0;
                double sum_x2 = 0, sum_y2 = 0;
                
                for (int k = 0; k < rows; k++) {
                    double x = features[k * cols + i];
                    double y = features[k * cols + j];
                    
                    sum_x += x;
                    sum_y += y;
                    sum_xy += x * y;
                    sum_x2 += x * x;
                    sum_y2 += y * y;
                }
                
                double numerator = rows * sum_xy - sum_x * sum_y;
                double denominator = std::sqrt((rows * sum_x2 - sum_x * sum_x) * 
                                             (rows * sum_y2 - sum_y * sum_y));
                
                corr_matrix[i * cols + j] = numerator / denominator;
            }
        }
    }
}
