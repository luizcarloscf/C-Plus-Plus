/**
 * @file
 * @brief Implementation of [Decision Tree
 * algorithm](https://en.wikipedia.org/wiki/Decision_tree)
 * @details Decision tree is a non-parametric supervised learning algorithm,
 * which can be utilized for both classification and regression tasks. This
 * implementation is only focused on its use for classification tasks. It has
 * a hierarchical, tree structure, which consists of a root node, branches,
 * internal nodes and leaf nodes. This algorithm utilizes Gini impurity to
 * identify the ideal attribute to split on. Gini impurity measures how often
 * a randomly chosen attribute is misclassified.
 * @author [Luiz Carlos Cosmi Filho](github.com/luizcarloscf)
 */
#include <algorithm>      /// for std::sort
#include <cassert>        /// for assert
#include <cmath>          /// for std::pow
#include <iostream>       /// for std::cout, std::endl
#include <memory>         /// for std::unique
#include <unordered_map>  /// for std::unordered_map
#include <vector>         /// for std::vector

/**
 * @namespace machine_learning
 * @brief Machine learning algorithms
 */
namespace machine_learning {

/**
 * @namespace decision_tree
 * @brief Decision tree algorithm
 */
namespace decision_tree {

/**
 * @brief Takes two vectors and zip element-wise into a vector of pairs.
 *
 * @tparam A typename of the first vector
 * @tparam B typename of the second vector
 * @param a first vector
 * @param b second vector
 * @param zipped vector of pairs
 */
template <typename C, typename D>
void zip(const std::vector<C>& a, const std::vector<D>& b,
         std::vector<std::pair<C, D>>& zipped) {
    for (size_t i = 0; i < a.size(); ++i) {
        zipped.push_back(std::make_pair(a[i], b[i]));
    }
}

/**
 * @brief Takes vector of pairs element-wise ziped and unzip into two
 * different vectors.
 *
 * @tparam A typename of the first vector
 * @tparam B typename of the second vector
 * @param zipped vector of pairs
 * @param a first vector
 * @param b second vector
 */
template <typename C, typename D>
void unzip(const std::vector<std::pair<C, D>>& zipped, std::vector<C>& a,
           std::vector<D>& b) {
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}

/**
 * @brief Decision Tree class using Gini impurity to find best split in dataset.
 *
 * @tparam A typename of attibutes/features vector
 * @tparam B typename of labels/output vector
 */
template <typename A, typename B>
class DecisionTree {
 private:
    double impurity;        ///< Dataset impurity
    int samples;            ///< Number of dataset samples
    int min_samples_split;  ///< Minimum number of samples required to split
    int max_depth;          ///< The maximum depth of the tree
    B prediction;           ///< Most frequent label of dataset
    std::unique_ptr<DecisionTree> left;   ///< Pointer to node on left
    std::unique_ptr<DecisionTree> right;  ///< Pointer to node on right
    std::pair<std::size_t, double>
        best_choice{};  ///< Best attribute and value to split

    /**
     * @brief Find most frequent value in vector.
     * @details Assumes that the vector is not empty.
     * @param b labels vector
     * @return std::pair<B, std::size_t> most frequent value and its frequency.
     */
    std::pair<B, std::size_t> most_frequent(const std::vector<B>& b) {
        std::unordered_map<B, int> frequency;
        for (auto i : b) {
            ++frequency[i];
        }
        std::pair<B, std::size_t> most_frequent;
        most_frequent.first = 0;
        most_frequent.second = 0;
        for (auto& kv : frequency) {
            if (kv.second > most_frequent.second) {
                most_frequent.second = kv.second;
                most_frequent.first = kv.first;
            }
        }
        return most_frequent;
    }

    /**
     * @brief Split dataset based on an attribute/feature value.
     * @param x attribute/feature dataset
     * @param y labels dataset
     * @param attribute attribute/feature to split
     * @param value value to split
     * @return std::tuple<std::vector<std::vector<A>>, std::vector<B>,
     * std::vector<std::vector<A>>, std::vector<B>> tuple with splitted
     * dataset
     */
    std::tuple<std::vector<std::vector<A>>, std::vector<B>,
               std::vector<std::vector<A>>, std::vector<B>>
    split(const std::vector<std::vector<A>>& x, const std::vector<B>& y,
          int attribute, int value) {
        std::vector<std::vector<A>> x_left{};
        std::vector<std::vector<A>> x_right{};
        std::vector<B> y_left{};
        std::vector<B> y_right{};
        for (size_t i = 0; i < y.size(); ++i) {
            if (x[i][attribute] < value) {
                y_left.push_back(y[i]);
                x_left.push_back(x[i]);
            } else {
                y_right.push_back(y[i]);
                x_right.push_back(x[i]);
            }
        }
        return std::make_tuple(x_left, y_left, x_right, y_right);
    }

    /**
     * @brief Get the impurity value of labels vector.
     *
     * @param y labels vector
     * @return double Gini impurity
     */
    double get_impurity(const std::vector<B>& y) {
        double impurity = 1.0;
        if (y.size() == 0)
            return 0.0;
        std::unordered_map<B, double> frequency;
        for (auto i : y) {
            ++frequency[i];
        }
        for (auto& kv : frequency) {
            impurity -= std::pow((kv.second / y.size()), 2);
        }
        return impurity;
    }

 public:
    /**
     * @brief Construct a new DecisionTree object
     *
     */
    DecisionTree(int max_depth, int min_samples_split)
        : max_depth(max_depth),
          min_samples_split(min_samples_split),
          left(nullptr),
          right(nullptr),
          impurity(0),
          samples(0),
          prediction(0){};

    /**
     * @brief Copy constructor for DecisionTree object.
     *
     * @param model object to be copied
     */
    DecisionTree(const DecisionTree& model) = default;

    /**
     * @brief Copy assignment operator for DecisionTree object.
     *
     * @param model object to be copied
     * @return DecisionTree&
     */
    DecisionTree& operator=(const DecisionTree& model) = default;

    /**
     * @brief Move constructor for DecisionTree object.
     */
    DecisionTree(DecisionTree&&) noexcept = default;

    /**
     * @brief Move assignment operator for DecisionTree object.
     *
     * @return DecisionTree&
     */
    DecisionTree& operator=(DecisionTree&&) noexcept = default;

    /**
     * @brief Destroy the DecisionTree object recursively.
     */
    ~DecisionTree() {
        if ((this->left != nullptr) && (this->right != nullptr)) {
            this->left.reset();
            this->right.reset();
        } else if (this->right != nullptr) {
            this->right.reset();
        } else if (this->left != nullptr) {
            this->left.reset();
        }
    }

    /**
     * @brief Build a decision tree classifier from the training set.
     *
     * @param X attributes/Features vector
     * @param Y labels/output vector
     */
    void fit(std::vector<std::vector<A>> X, std::vector<B> Y) {
        this->impurity = this->get_impurity(Y);
        if (this->max_depth == 0) {
            auto predict = this->most_frequent(Y);
            this->prediction = predict.first;
            return;
        }
        double gini = 0.0;
        std::vector<std::pair<std::vector<A>, B>> zipped;
        zip(X, Y, zipped);
        for (std::size_t i = 0; i < X[0].size(); i++) {
            std::sort(std::begin(zipped), std::end(zipped),
                      [&](const std::pair<std::vector<A>, B>& a,
                          const std::pair<std::vector<A>, B>& b) {
                          return a.first[i] < b.first[i];
                      });
            unzip(zipped, X, Y);
            for (std::size_t j = 1; j < Y.size(); j++) {
                auto value_before = X[j - 1][i];
                auto value = X[j][i];
                double split_value = static_cast<double>(value - value_before);
                split_value = (split_value / 2) + value_before;
                auto result = this->split(X, Y, i, split_value);
                auto x_left_ = std::get<0>(result);
                auto y_left_ = std::get<1>(result);
                auto x_right_ = std::get<2>(result);
                auto y_right_ = std::get<3>(result);
                if ((x_left_.empty()) || (x_right_.empty())) {
                    continue;
                } else if ((x_left_.size() < this->min_samples_split) ||
                           (x_right_.size() < this->min_samples_split)) {
                    continue;
                } else {
                    auto impurity_left = this->get_impurity(y_left_);
                    auto impurity_right = this->get_impurity(y_right_);
                    auto impurity_total =
                        ((impurity_left * y_left_.size()) / Y.size()) +
                        ((impurity_right * y_right_.size()) / Y.size());
                    auto gini_current = this->impurity - impurity_total;
                    if (gini_current > gini) {
                        gini = gini_current;
                        this->best_choice.first = i;
                        this->best_choice.second = split_value;
                    }
                }
            }
        }
        if (gini > 0.0) {
            auto best_result = this->split(X, Y, this->best_choice.first,
                                           this->best_choice.second);
            auto x_left = std::get<0>(best_result);
            auto y_left = std::get<1>(best_result);
            auto x_right = std::get<2>(best_result);
            auto y_right = std::get<3>(best_result);
            this->left = std::unique_ptr<DecisionTree>(
                new DecisionTree(this->max_depth - 1, this->min_samples_split));
            this->right = std::unique_ptr<DecisionTree>(
                new DecisionTree(this->max_depth - 1, this->min_samples_split));
            this->left->fit(x_left, y_left);
            this->right->fit(x_right, y_right);
        } else {
            auto counts = this->most_frequent(Y);
            this->prediction = counts.first;
            return;
        }
    }

    /**
     * @brief Predict class value for a sample.
     *
     * @param x sample attribute/feature to perform inference
     * @return B predicted value
     */
    B predict(const std::vector<A>& x) {
        if ((this->left == nullptr) || (this->right == nullptr)) {
            return this->prediction;
        } else {
            if (x[this->best_choice.first] < this->best_choice.second) {
                return this->left->predict(x);
            } else {
                return this->right->predict(x);
            }
        }
    }
};

}  // namespace decision_tree
}  // namespace machine_learning

/**
 * @brief Self-test implementations
 * @returns void
 */
static void test() {
    std::cout << "Test 1..." << std::endl;
    auto model1 =
        machine_learning::decision_tree::DecisionTree<double, size_t>(5, 4);
    std::vector<std::vector<double>> X1 = {{0}, {0}, {1}, {1},
                                           {2}, {2}, {3}, {3}};
    std::vector<size_t> Y1 = {2, 2, 2, 2, 3, 3, 3, 3};
    model1.fit(X1, Y1);
    std::vector<double> Z11 = {0};
    std::vector<double> Z12 = {1};
    std::vector<double> Z13 = {2};
    std::vector<double> Z14 = {3};
    std::cout << "Predicted: " << model1.predict(Z11) << std::endl;
    assert(model1.predict(Z11) == 2);
    std::cout << "Predicted: " << model1.predict(Z12) << std::endl;
    assert(model1.predict(Z12) == 2);
    std::cout << "Predicted: " << model1.predict(Z13) << std::endl;
    assert(model1.predict(Z13) == 3);
    std::cout << "Predicted: " << model1.predict(Z14) << std::endl;
    assert(model1.predict(Z14) == 3);
    std::cout << "Test 1... DONE!" << std::endl;
    std::cout << "Test 2..." << std::endl;
    auto model2 =
        machine_learning::decision_tree::DecisionTree<int, int>(10, 1);
    std::vector<std::vector<int>> X2 = {{2, 4, 11}, {1, 40, 14}, {41, 0, 0},
                                        {40, 0, 0}, {40, 4, 0},  {41, 5, 0}};
    std::vector<int> Y2 = {0, 0, 1, 1, 3, 3};
    model2.fit(X2, Y2);
    std::vector<int> Z21 = {0, 6, 40};
    std::vector<int> Z22 = {50, 0, 40};
    std::vector<int> Z23 = {51, 6, 40};
    std::cout << "Predicted: " << model2.predict(Z21) << std::endl;
    assert(model2.predict(Z21) == 0);
    std::cout << "Predicted: " << model2.predict(Z22) << std::endl;
    assert(model2.predict(Z22) == 1);
    std::cout << "Predicted: " << model2.predict(Z23) << std::endl;
    assert(model2.predict(Z23) == 3);
    std::cout << "Test 2... DONE!" << std::endl;
}

/**
 * @brief Main function
 * @param argc commandline argument count (ignored)
 * @param argv commandline array of arguments (ignored)
 * @return int 0 on exit
 */
int main(int argc, char* argv[]) {
    test();  // run self-test implementations
    return 0;
}