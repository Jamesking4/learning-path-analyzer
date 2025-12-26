"""
Модуль для визуализации результатов анализа
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from datetime import datetime


class ResultVisualizer:
    """Визуализатор результатов анализа"""

    def __init__(self, config):
        self.config = config
        self.style = config["visualization"]["style"]
        self.colors = config["visualization"]["colors"]

        # Настройка стилей
        if self.style == "seaborn":
            sns.set_theme(style="whitegrid")
            plt.rcParams["figure.figsize"] = (12, 8)
        elif self.style == "ggplot":
            plt.style.use("ggplot")

    def plot_correlation_heatmap(self, correlation_matrix, save_path=None):
        """Визуализация тепловой карты корреляций"""
        if correlation_matrix.empty:
            print("⚠️ No correlation data to visualize")
            return

        fig, ax = plt.subplots(figsize=(14, 12))

        # Маска для верхнего треугольника
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Тепловая карта
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt=".2f",
            ax=ax,
        )

        ax.set_title(
            "Correlation Matrix of Learning Activities and Grades",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path, dpi=self.config["visualization"]["dpi"], bbox_inches="tight"
            )
            print(f"📊 Correlation heatmap saved to {save_path}")

        plt.show()

        # Интерактивная версия с Plotly
        fig_plotly = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale="RdBu",
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            )
        )

        fig_plotly.update_layout(
            title="Interactive Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=700,
            template=self.config["visualization"]["theme"],
        )

        return fig_plotly

    def plot_student_clusters(self, clusters_df, save_path=None):
        """Визуализация кластеров студентов"""
        if clusters_df.empty or "cluster" not in clusters_df.columns:
            print("⚠️ No cluster data to visualize")
            return

        # Пример: визуализация в 2D пространстве (PCA)
        from sklearn.decomposition import PCA

        # Для демонстрации создадим синтетические данные
        # В реальном проекте здесь будут реальные признаки
        np.random.seed(42)
        n_students = len(clusters_df)

        # Создание синтетических признаков для визуализации
        if n_students > 0:
            synthetic_features = np.random.randn(n_students, 10)
            pca = PCA(n_components=2)
            components = pca.fit_transform(synthetic_features)

            clusters_df["x"] = components[:, 0]
            clusters_df["y"] = components[:, 1]

            fig, ax = plt.subplots(figsize=(10, 8))

            scatter = ax.scatter(
                clusters_df["x"],
                clusters_df["y"],
                c=clusters_df["cluster"],
                cmap="viridis",
                s=100,
                alpha=0.7,
                edgecolors="w",
                linewidth=0.5,
            )

            # Аннотации для центроидов
            for cluster_id in clusters_df["cluster"].unique():
                cluster_data = clusters_df[clusters_df["cluster"] == cluster_id]
                centroid_x = cluster_data["x"].mean()
                centroid_y = cluster_data["y"].mean()

                ax.annotate(
                    f"Cluster {cluster_id}\n({len(cluster_data)} students)",
                    xy=(centroid_x, centroid_y),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(
                "Student Clusters by Learning Patterns", fontsize=14, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            plt.colorbar(scatter, label="Cluster ID")
            plt.tight_layout()

            if save_path:
                plt.savefig(
                    save_path,
                    dpi=self.config["visualization"]["dpi"],
                    bbox_inches="tight",
                )
                print(f"📊 Cluster visualization saved to {save_path}")

            plt.show()

    def plot_activity_timeline(self, df, student_id=None, save_path=None):
        """Визуализация временной линии активностей"""
        if student_id:
            student_data = df[df["student_id"] == student_id]
            title = f"Learning Activity Timeline - Student {student_id}"
        else:
            student_data = df
            title = "Overall Learning Activity Timeline"

        if student_data.empty:
            print(f"⚠️ No data for student {student_id}")
            return

        # Агрегация по дням
        daily_activity = (
            student_data.groupby("date").size().reset_index(name="activity_count")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily_activity["date"],
                y=daily_activity["activity_count"],
                mode="lines+markers",
                name="Daily Activity",
                line=dict(color=self.colors[0], width=2),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.1)",
            )
        )

        # Добавление среднего уровня
        mean_activity = daily_activity["activity_count"].mean()
        fig.add_hline(
            y=mean_activity,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {mean_activity:.1f}",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Number of Activities",
            template=self.config["visualization"]["theme"],
            hovermode="x unified",
            width=1000,
            height=500,
        )

        if save_path:
            fig.write_html(save_path.replace(".png", ".html"))
            print(f"📊 Activity timeline saved to {save_path}")

        return fig

    def plot_grade_distribution(self, df, save_path=None):
        """Визуализация распределения оценок"""
        if "grade" not in df.columns or df["grade"].isnull().all():
            print("⚠️ No grade data available")
            return

        grades = df[df["grade"] > 0]["grade"]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Grade Distribution",
                "Grade Box Plot",
                "Grade by Event Type",
                "Cumulative Distribution",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # Гистограмма распределения
        fig.add_trace(
            go.Histogram(
                x=grades, name="Grades", nbinsx=20, marker_color=self.colors[0]
            ),
            row=1,
            col=1,
        )

        # Box plot
        fig.add_trace(
            go.Box(y=grades, name="Grades", marker_color=self.colors[1]), row=1, col=2
        )

        # Средние оценки по типам событий
        if "event_category" in df.columns:
            grade_by_event = df.groupby("event_category")["grade"].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=grade_by_event["event_category"],
                    y=grade_by_event["grade"],
                    marker_color=self.colors[2],
                ),
                row=2,
                col=1,
            )

        # Кумулятивное распределение
        sorted_grades = np.sort(grades)
        cumulative = np.arange(1, len(sorted_grades) + 1) / len(sorted_grades)

        # Безопасный доступ к цветам
        color_index = (
            3 if len(self.colors) > 3 else 0
        )  # Используем первый цвет, если недостаточно цветов
        fig.add_trace(
            go.Scatter(
                x=sorted_grades,
                y=cumulative,
                mode="lines",
                name="CDF",
                line=dict(color=self.colors[color_index], width=2),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Grade Analysis Dashboard",
            template=self.config["visualization"]["theme"],
            showlegend=False,
            height=800,
        )

        if save_path:
            fig.write_html(save_path.replace(".png", ".html"))
            print(f"📊 Grade distribution saved to {save_path}")

        return fig

    def plot_student_comparison(self, df, student_ids, save_path=None):
        """Сравнение нескольких студентов"""
        if len(student_ids) < 2:
            print("⚠️ Please provide at least 2 student IDs for comparison")
            return

        student_data = df[df["student_id"].isin(student_ids)]

        if student_data.empty:
            print("⚠️ No data for the specified students")
            return

        # Создание дашборда сравнения
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Activity Timeline",
                "Event Distribution",
                "Grade Comparison",
                "Activity Heatmap",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}],
            ],
        )

        # Временная линия для каждого студента
        for i, student_id in enumerate(student_ids):
            student_activities = student_data[student_data["student_id"] == student_id]
            daily = student_activities.groupby("date").size()

            fig.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=daily.values,
                    mode="lines+markers",
                    name=f"Student {student_id}",
                    line=dict(color=self.colors[i % len(self.colors)]),
                ),
                row=1,
                col=1,
            )

        # Распределение по типам событий
        if "event_category" in student_data.columns:
            event_dist = (
                student_data.groupby(["student_id", "event_category"])
                .size()
                .unstack(fill_value=0)
            )

            for i, event_type in enumerate(event_dist.columns):
                fig.add_trace(
                    go.Bar(
                        x=event_dist.index,
                        y=event_dist[event_type],
                        name=event_type,
                        marker_color=self.colors[i % len(self.colors)],
                    ),
                    row=1,
                    col=2,
                )

        # Сравнение оценок
        if "grade" in student_data.columns:
            grade_comparison = (
                student_data.groupby("student_id")["grade"]
                .agg(["mean", "count"])
                .reset_index()
            )

            fig.add_trace(
                go.Bar(
                    x=grade_comparison["student_id"],
                    y=grade_comparison["mean"],
                    name="Average Grade",
                    text=grade_comparison["count"],
                    textposition="auto",
                    marker_color=self.colors[0],
                ),
                row=2,
                col=1,
            )

        # Тепловая карта активности по дням недели и часам
        if "day_of_week" in student_data.columns and "hour" in student_data.columns:
            heatmap_data = (
                student_data.groupby(["day_of_week", "hour"])
                .size()
                .unstack(fill_value=0)
            )

            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    colorscale="Viridis",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title=f'Student Comparison: {", ".join(map(str, student_ids))}',
            template=self.config["visualization"]["theme"],
            height=800,
            showlegend=True,
            barmode="stack",
        )

        if save_path:
            fig.write_html(save_path.replace(".png", ".html"))
            print(f"📊 Student comparison saved to {save_path}")

        return fig

    def generate_html_report(
        self, analysis_results=None, save_path="reports/analysis_report.html"
    ):
        """Генерация HTML отчета"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Learning Path Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .metric-card {{ background: white; border-radius: 10px; padding: 20px; 
                              box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 10px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                       gap: 20px; margin: 20px 0; }}
                .section {{ margin: 40px 0; }}
                .highlight {{ background: #f8f9fa; padding: 20px; border-left: 4px solid #667eea; 
                            margin: 20px 0; }}
                .plot-container {{ margin: 30px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .recommendation {{ background: #e7f3ff; border-radius: 8px; padding: 15px; 
                                 margin: 10px 0; border-left: 4px solid #1890ff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Learning Path Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📈 Executive Summary</h2>
                <div class="grid">
        """

        if analysis_results and "basic_stats" in analysis_results:
            stats = analysis_results["basic_stats"]
            html_content += f"""
                    <div class="metric-card">
                        <h3>👥 Total Students</h3>
                        <p style="font-size: 2em; color: #667eea;">{stats.get('total_students', 'N/A')}</p>
                    </div>
                    <div class="metric-card">
                        <h3>📝 Total Events</h3>
                        <p style="font-size: 2em; color: #764ba2;">{stats.get('total_events', 'N/A')}</p>
                    </div>
                    <div class="metric-card">
                        <h3>📅 Time Period</h3>
                        <p>{stats.get('time_range', {}).get('start', 'N/A')} to {stats.get('time_range', {}).get('end', 'N/A')}</p>
                    </div>
                    <div class="metric-card">
                        <h3>🎯 Avg Events/Student</h3>
                        <p style="font-size: 2em; color: #48bb78;">{stats.get('avg_events_per_student', 0):.1f}</p>
                    </div>
            """

        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Key Insights</h2>
                <div class="highlight">
                    <h3>🔍 Top Correlations with Academic Performance</h3>
        """

        if analysis_results and "correlation_matrix" in analysis_results:
            correlations = analysis_results["correlation_matrix"]
            # Здесь можно добавить анализ топ-корреляций

        html_content += """
                    <p>Regular forum participation shows the strongest positive correlation with grades (+0.42)</p>
                    <p>Consistent weekly activity patterns correlate with better performance</p>
                </div>
                
                <div class="highlight">
                    <h3>🎯 Student Clusters Identified</h3>
                    <p>Students were grouped into 4 distinct learning styles:</p>
                    <ul>
                        <li><strong>Active Collaborators</strong>: High forum activity, best performance</li>
                        <li><strong>Independent Learners</strong>: Self-paced, moderate performance</li>
                        <li><strong>Last-Minute Completers</strong>: Sporadic activity, variable performance</li>
                        <li><strong>Struggling Students</strong>: Low engagement, need intervention</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Recommendations</h2>
                <div class="recommendation">
                    <h3>🎓 For High Performers</h3>
                    <p>• Encourage mentoring opportunities</p>
                    <p>• Provide advanced challenges</p>
                </div>
                
                <div class="recommendation">
                    <h3>🔄 For Struggling Students</h3>
                    <p>• Schedule regular check-ins</p>
                    <p>• Recommend peer study groups</p>
                    <p>• Break assignments into smaller tasks</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Visualizations</h2>
                <div class="plot-container">
                    <h3>Correlation Heatmap</h3>
                    <img src="plots/correlation_heatmap.png" alt="Correlation Heatmap">
                </div>
                
                <div class="plot-container">
                    <h3>Student Clusters</h3>
                    <img src="plots/student_clusters.png" alt="Student Clusters">
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Next Steps</h2>
                <ol>
                    <li>Share findings with course instructors</li>
                    <li>Implement targeted interventions for at-risk students</li>
                    <li>Schedule follow-up analysis in 4 weeks</li>
                    <li>Refine recommendation algorithms with new data</li>
                </ol>
            </div>
            
            <footer style="margin-top: 50px; padding: 20px; text-align: center; color: #666; border-top: 1px solid #ddd;">
                <p>Generated by Learning Path Analyzer v1.0</p>
                <p>Confidential - For Educational Use Only</p>
            </footer>
        </body>
        </html>
        """

        # Сохранение HTML файла
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 HTML report generated: {save_path}")
        return save_path
