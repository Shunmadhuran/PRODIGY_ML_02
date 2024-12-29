import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def select_file():
    global filename
    filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    file_path_label.config(text=filename)

def cluster_data():
    try:
        # Load data
        data = pd.read_csv(filename)

        # Get selected features
        selected_features = [feature for feature, selected in feature_checkboxes.items() if selected.get()]
        if not selected_features:
            raise ValueError("Please select at least one feature.")

        # Preprocessing
        data_for_clustering = data[selected_features]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)

        # Get number of clusters
        try:
            n_clusters = int(num_clusters_entry.get())
        except ValueError:
            raise ValueError("Please enter a valid number of clusters.")

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(scaled_data)

        # Calculate CLV (simplified)
        data['CLV'] = data['total_spent'] * (data['number_of_purchases'] / 100)

        # Cluster analysis
        cluster_summary = data.groupby('cluster').agg(
            avg_spent=('total_spent', 'mean'),
            avg_purchases=('number_of_purchases', 'mean'),
            avg_age=('age', 'mean'),
            avg_clv=('CLV', 'mean'),
            count=('customer_id', 'count')
        ).reset_index()

        # Update cluster summary table
        tree.delete(*tree.get_children())
        for index, row in cluster_summary.iterrows():
            tree.insert("", "end", values=[row['cluster']] + list(row[1:]))

        # Clear previous plot
        for widget in plot_frame.winfo_children():
            widget.destroy()

        # Plot customer segments
        fig, ax = plt.subplots(figsize=(8, 4))  # Create a new figure
        scatter = ax.scatter(
            data['total_spent'],
            data['number_of_purchases'],
            c=data['cluster'],
            cmap='viridis',
            alpha=0.6
        )
        ax.set_xlabel('Total Spent', fontsize=10)
        ax.set_ylabel('Number of Purchases', fontsize=10)
        ax.set_title('Customer Segmentation')
        ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels
        fig.subplots_adjust(bottom=0.3)  # Adjust bottom margin for visibility
        fig.colorbar(scatter, ax=ax, label='Cluster')

        # Embed the new plot into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        error_label.config(text=f"An error occurred: {str(e)}")


# GUI setup
root = tk.Tk()
root.title("Customer Segmentation")

file_path_label = tk.Label(root, text="No file selected")
file_path_label.pack()

select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack()

feature_checkboxes = {
    'total_spent': tk.IntVar(),
    'number_of_purchases': tk.IntVar(),
    'age': tk.IntVar()
}
for feature, var in feature_checkboxes.items():
    checkbutton = tk.Checkbutton(root, text=feature, variable=var)
    checkbutton.pack()

num_clusters_label = tk.Label(root, text="Number of Clusters:")
num_clusters_label.pack()
num_clusters_entry = tk.Entry(root)
num_clusters_entry.pack()

cluster_button = tk.Button(root, text="Cluster", command=cluster_data)
cluster_button.pack()

error_label = tk.Label(root, text="", fg="red")
error_label.pack()

# Cluster summary table
tree_frame = tk.Frame(root)
tree_frame.pack(fill=tk.BOTH, expand=1)
tree_scroll = tk.Scrollbar(tree_frame)
tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
tree = ttk.Treeview(tree_frame, columns=("Cluster", "Avg Spent", "Avg Purchases", "Avg Age", "Avg CLV", "Count"), yscrollcommand=tree_scroll.set)
tree.heading("Cluster", text="Cluster")
tree.heading("Avg Spent", text="Avg Spent")
tree.heading("Avg Purchases", text="Avg Purchases")
tree.heading("Avg Age", text="Avg Age")
tree.heading("Avg CLV", text="Avg CLV")
tree.heading("Count", text="Count")
tree.pack(side=tk.LEFT, fill=tk.BOTH)
tree_scroll.config(command=tree.yview)

# Plot frame
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=1)

root.mainloop()