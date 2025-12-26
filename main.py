#!/usr/bin/env python3
"""
Основной скрипт для запуска анализа образовательных траекторий
"""

import argparse
import sys
import os
from datetime import datetime

# Добавляем путь к src в sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_parser import LogParser
from analyzer import LearningAnalyzer
from visualizer import ResultVisualizer
from recommender import RecommendationEngine
import yaml
import pandas as pd


def load_config(config_path="config.yaml"):
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Learning Path Analyzer')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-o', '--output', default='reports', help='Output directory')
    parser.add_argument('-c', '--config', default='config.yaml', help='Config file path')
    parser.add_argument('--student-id', help='Analyze specific student ID')
    parser.add_argument('--timeframe', help='Timeframe filter (YYYY-MM)')
    parser.add_argument('--min-grade', type=float, help='Minimum grade threshold')
    parser.add_argument('--export-json', action='store_true', help='Export results as JSON')
    parser.add_argument('--visualize-only', action='store_true', help='Only generate visualizations')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Создание выходных директорий
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)
    
    print(f"🚀 Starting Learning Path Analyzer")
    print(f"📊 Input file: {args.input}")
    print(f"📁 Output directory: {args.output}")
    
    if not args.visualize_only:
        # 1. Парсинг данных
        print("\n📈 Step 1: Parsing data...")
        parser = LogParser(config)
        df = parser.parse_csv(args.input)
        
        if args.timeframe:
            df = parser.filter_by_timeframe(df, args.timeframe)
        
        # 2. Анализ данных
        print("📊 Step 2: Analyzing data...")
        analyzer = LearningAnalyzer(config)
        
        # Базовые метрики
        basic_stats = analyzer.calculate_basic_metrics(df)
        print(f"   Total students: {basic_stats['total_students']}")
        print(f"   Total events: {basic_stats['total_events']}")
        print(f"   Time range: {basic_stats['time_range']}")
        
        # Расчет корреляций
        correlation_matrix = analyzer.calculate_correlations(df)
        
        # Кластеризация студентов
        print("   Clustering students...")
        clusters = analyzer.cluster_students(df)
        
        # Анализ временных паттернов
        print("   Analyzing temporal patterns...")
        time_patterns = analyzer.analyze_time_patterns(df)
        
        # 3. Генерация рекомендаций
        print("🎯 Step 3: Generating recommendations...")
        recommender = RecommendationEngine(config)
        
        if args.student_id:
            # Персонализированные рекомендации
            student_recommendations = recommender.generate_personalized_recommendations(
                df, args.student_id
            )
            print(f"\n📋 Recommendations for student {args.student_id}:")
            for i, rec in enumerate(student_recommendations[:5], 1):
                print(f"   {i}. {rec}")
            
            # Сохранение рекомендаций
            recommender.save_recommendations(
                {args.student_id: student_recommendations},
                os.path.join(args.output, f"recommendations_{args.student_id}.json")
            )
        else:
            # Общие рекомендации
            general_recommendations = recommender.generate_general_recommendations(df)
            print(f"\n📋 General recommendations:")
            for i, rec in enumerate(general_recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        # 4. Сохранение результатов анализа
        print("\n💾 Step 4: Saving analysis results...")
        results = {
            'basic_stats': basic_stats,
            'correlation_matrix': correlation_matrix.to_dict(),
            'clusters': clusters.to_dict(),
            'time_patterns': time_patterns,
            'timestamp': datetime.now().isoformat()
        }
        
        analyzer.save_results(results, os.path.join(args.output, 'analysis_results.json'))
        
        if args.export_json:
            df.to_json(os.path.join(args.output, 'processed_data.json'), orient='records')
    
    # 5. Визуализация результатов
    print("\n🎨 Step 5: Creating visualizations...")
    visualizer = ResultVisualizer(config)
    
    if not args.visualize_only:
        # Загружаем сохраненные результаты
        results_path = os.path.join(args.output, 'analysis_results.json')
        if os.path.exists(results_path):
            import json
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Визуализация корреляций
            visualizer.plot_correlation_heatmap(
                pd.DataFrame(results['correlation_matrix']),
                save_path=os.path.join(args.output, 'plots', 'correlation_heatmap.png')
            )
            
            # Визуализация кластеров
            visualizer.plot_student_clusters(
                pd.DataFrame(results['clusters']),
                save_path=os.path.join(args.output, 'plots', 'student_clusters.png')
            )
    
    # Создание HTML отчета
    print("📄 Generating HTML report...")
    report_path = visualizer.generate_html_report(
        results if not args.visualize_only else None,
        save_path=os.path.join(args.output, 'analysis_report.html')
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved in: {args.output}")
    print(f"📄 Report available at: {report_path}")
    
    if config['report']['auto_open_browser'] and not args.visualize_only:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()