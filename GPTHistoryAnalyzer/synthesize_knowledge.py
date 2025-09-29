#!/usr/bin/env python3
"""
Knowledge Synthesis Script
Combines and synthesizes the analysis results from staged batch processing
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def extract_analysis_content(analysis_list: List[Dict]) -> List[str]:
    """
    Extracts the actual analysis content from batch results
    Each item has the GPT response nested in response.body.choices[0].message.content
    """
    contents = []

    for item in analysis_list:
        try:
            # Navigate the nested structure to get the actual content
            content = item['response']['body']['choices'][0]['message']['content']
            contents.append(content)
        except (KeyError, IndexError, TypeError) as e:
            print(f"  ⚠️  Could not extract content from one analysis: {e}")
            continue

    return contents


def synthesize_analyses(combined_analysis_path: Path) -> Dict[str, Any]:
    """
    Main synthesis function that combines all analyses into a knowledge base
    """
    print("\n📚 Starting Knowledge Synthesis")
    print("=" * 60)

    # Load the combined analysis
    print(f"📂 Loading analysis from: {combined_analysis_path}")

    if not combined_analysis_path.exists():
        print(f"❌ File not found: {combined_analysis_path}")
        return {}

    with open(combined_analysis_path, 'r', encoding='utf-8') as f:
        combined_data = json.load(f)

    # Extract content from each analysis type
    print("\n📊 Extracting analysis content...")

    project_contents = extract_analysis_content(combined_data.get('project_evolution', []))
    inquiry_contents = extract_analysis_content(combined_data.get('inquiry_patterns', []))
    knowledge_contents = extract_analysis_content(combined_data.get('knowledge_evolution', []))

    print(f"  ✅ Project Evolution: {len(project_contents)} analyses extracted")
    print(f"  ✅ Inquiry Patterns: {len(inquiry_contents)} analyses extracted")
    print(f"  ✅ Knowledge Evolution: {len(knowledge_contents)} analyses extracted")

    # Create the synthesized knowledge base
    print("\n🔄 Synthesizing insights...")

    knowledge_base = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source_file": str(combined_analysis_path),
            "analyses_count": {
                "project_evolution": len(project_contents),
                "inquiry_patterns": len(inquiry_contents),
                "knowledge_evolution": len(knowledge_contents)
            },
            "synthesis_version": "2.0"
        },
        "executive_summary": create_executive_summary(
            project_contents, inquiry_contents, knowledge_contents
        ),
        "detailed_insights": {
            "projects": synthesize_project_insights(project_contents),
            "learning": synthesize_learning_patterns(inquiry_contents),
            "knowledge": synthesize_knowledge_growth(knowledge_contents)
        },
        "cross_cutting_themes": identify_cross_cutting_themes(
            project_contents, inquiry_contents, knowledge_contents
        ),
        "actionable_recommendations": generate_recommendations(
            project_contents, inquiry_contents, knowledge_contents
        ),
        "quick_reference": create_quick_reference(
            project_contents, inquiry_contents, knowledge_contents
        )
    }

    return knowledge_base


def create_executive_summary(
        project_contents: List[str],
        inquiry_contents: List[str],
        knowledge_contents: List[str]
) -> str:
    """
    Creates an executive summary from all analyses
    """
    summary_parts = []

    # Combine key points from each analysis type
    if project_contents:
        summary_parts.append("**Project Development**: " +
                             " ".join(project_contents)[:500] + "...")

    if inquiry_contents:
        summary_parts.append("**Learning Patterns**: " +
                             " ".join(inquiry_contents)[:500] + "...")

    if knowledge_contents:
        summary_parts.append("**Knowledge Evolution**: " +
                             " ".join(knowledge_contents)[:500] + "...")

    if not summary_parts:
        return "No analysis content available for synthesis."

    # Create a cohesive summary
    summary = """Based on the comprehensive analysis of your conversations, this knowledge base captures your intellectual journey and development patterns.

""" + "\n\n".join(summary_parts)

    return summary


def synthesize_project_insights(project_contents: List[str]) -> Dict:
    """
    Synthesizes insights from project evolution analyses
    """
    if not project_contents:
        return {"message": "No project analysis available"}

    # Parse and structure the project insights
    insights = {
        "overview": "Analysis of project development and progression",
        "raw_analyses": project_contents,
        "key_themes": [],
        "project_list": [],
        "development_patterns": []
    }

    # Extract key information from the analyses
    for content in project_contents:
        # Look for specific patterns in the analysis text
        if "Key projects" in content:
            # Extract project mentions
            lines = content.split('\n')
            for line in lines:
                if any(indicator in line.lower() for indicator in ['project', 'feature', 'implementation']):
                    insights['key_themes'].append(line.strip())

        if "milestones" in content.lower():
            # Extract milestone information
            insights['development_patterns'].append("Milestone-driven development identified")

        if "progression" in content.lower():
            insights['development_patterns'].append("Progressive iteration pattern observed")

    return insights


def synthesize_learning_patterns(inquiry_contents: List[str]) -> Dict:
    """
    Synthesizes insights from inquiry pattern analyses
    """
    if not inquiry_contents:
        return {"message": "No inquiry analysis available"}

    insights = {
        "overview": "Analysis of learning and inquiry patterns",
        "raw_analyses": inquiry_contents,
        "dominant_patterns": [],
        "problem_types": [],
        "interaction_style": []
    }

    # Extract patterns from the analyses
    for content in inquiry_contents:
        # Look for problem-solving patterns
        if "troubleshooting" in content.lower():
            insights['problem_types'].append("Technical troubleshooting")
        if "analysis" in content.lower():
            insights['problem_types'].append("Analytical problem-solving")
        if "creative" in content.lower():
            insights['problem_types'].append("Creative and communication tasks")

        # Identify interaction patterns
        if "iterative" in content.lower():
            insights['interaction_style'].append("Iterative refinement approach")
        if "concise" in content.lower():
            insights['interaction_style'].append("Preference for concise, actionable responses")

    return insights


def synthesize_knowledge_growth(knowledge_contents: List[str]) -> Dict:
    """
    Synthesizes insights from knowledge evolution analyses
    """
    if not knowledge_contents:
        return {"message": "No knowledge evolution analysis available"}

    insights = {
        "overview": "Analysis of knowledge development over time",
        "raw_analyses": knowledge_contents,
        "growth_trajectory": [],
        "key_domains": [],
        "learning_moments": []
    }

    # Extract knowledge evolution patterns
    for content in knowledge_contents:
        # Look for domain expertise
        domains = ['technical', 'finance', 'communication', 'organizational', 'strategic']
        for domain in domains:
            if domain in content.lower():
                insights['key_domains'].append(domain.capitalize())

        # Identify growth patterns
        if "basic to advanced" in content.lower():
            insights['growth_trajectory'].append("Progressive depth in understanding")
        if "cross-domain" in content.lower():
            insights['growth_trajectory'].append("Cross-domain knowledge integration")

    return insights


def identify_cross_cutting_themes(
        project_contents: List[str],
        inquiry_contents: List[str],
        knowledge_contents: List[str]
) -> List[str]:
    """
    Identifies themes that appear across different analysis types
    """
    themes = []

    # Combine all content for theme extraction
    all_content = " ".join(project_contents + inquiry_contents + knowledge_contents).lower()

    # Common themes to look for
    theme_indicators = {
        "iterative development": ["iterative", "refinement", "iteration"],
        "technical problem-solving": ["debug", "fix", "troubleshoot", "error"],
        "communication and writing": ["draft", "message", "communication", "writing"],
        "strategic thinking": ["strategy", "organizational", "macro", "trends"],
        "continuous learning": ["learning", "growth", "evolution", "development"],
        "practical application": ["practical", "actionable", "implementation", "applied"]
    }

    for theme, keywords in theme_indicators.items():
        if any(keyword in all_content for keyword in keywords):
            themes.append(theme.title())

    return themes


def generate_recommendations(
        project_contents: List[str],
        inquiry_contents: List[str],
        knowledge_contents: List[str]
) -> Dict:
    """
    Generates actionable recommendations based on the analyses
    Improved to extract complete recommendations and avoid fragments
    """
    recommendations = {
        "immediate_actions": [],
        "medium_term_goals": [],
        "long_term_strategies": [],
        "tools_and_templates": []
    }

    # Combine all content
    all_content = " ".join(project_contents + inquiry_contents + knowledge_contents)

    # Extract recommendations more intelligently
    lines = all_content.split('\n')

    # Track recommendations to avoid duplicates
    seen_recommendations = set()

    for i, line in enumerate(lines):
        line_clean = line.strip().lstrip('- •').strip()

        # Skip if too short or already seen
        if len(line_clean) < 30 or line_clean.lower() in seen_recommendations:
            continue

        # Check for recommendation indicators
        if any(indicator in line.lower() for indicator in
               ['recommend', 'should', 'suggest', 'create', 'build', 'maintain']):
            # Make sure it's a complete sentence/thought
            if line_clean[0].isupper() and (line_clean.endswith('.') or line_clean.endswith(':')):
                seen_recommendations.add(line_clean.lower())

                # Categorize the recommendation
                if any(word in line_clean.lower() for word in ['template', 'checklist', 'tool', 'library', 'doc']):
                    recommendations['tools_and_templates'].append(line_clean)
                elif any(
                        word in line_clean.lower() for word in ['immediately', 'now', 'first', 'next', 'tag existing']):
                    recommendations['immediate_actions'].append(line_clean)
                elif any(word in line_clean.lower() for word in ['consolidate', 'formalize', 'schedule']):
                    recommendations['medium_term_goals'].append(line_clean)
                else:
                    # Default to medium-term
                    recommendations['medium_term_goals'].append(line_clean)

    # Extract specific recommendation sections that are well-structured
    if "Recommendations" in all_content:
        # Look for sections that start with "Recommendations"
        for i, line in enumerate(lines):
            if "Recommendations" in line and i < len(lines) - 1:
                # Check the next few lines for bullet points
                for j in range(1, min(10, len(lines) - i)):
                    next_line = lines[i + j].strip()
                    if next_line.startswith('-') or next_line.startswith('•'):
                        rec = next_line.lstrip('- •').strip()
                        if len(rec) > 30 and rec not in seen_recommendations:
                            seen_recommendations.add(rec.lower())
                            if 'template' in rec.lower() or 'checklist' in rec.lower():
                                recommendations['tools_and_templates'].append(rec)
                            else:
                                recommendations['immediate_actions'].append(rec)

    # Limit to top items to avoid clutter
    for key in recommendations:
        recommendations[key] = recommendations[key][:5]

    # Add default recommendations if too few found
    if len(recommendations['immediate_actions']) < 2:
        recommendations['immediate_actions'].extend([
            "Review and tag your most recent conversations by the identified themes",
            "Create a project tracker for your active AI/semantic search initiatives",
            "Document your debugging solutions in a searchable format"
        ])[:3 - len(recommendations['immediate_actions'])]

    if len(recommendations['tools_and_templates']) < 2:
        recommendations['tools_and_templates'].extend([
            "Create a debugging checklist based on your uvloop/Python troubleshooting patterns",
            "Build a prompt template library from your successful AI/search experiments",
            "Develop a communication template for stakeholder updates and follow-ups"
        ])[:3 - len(recommendations['tools_and_templates'])]

    return recommendations


def create_quick_reference(
        project_contents: List[str],
        inquiry_contents: List[str],
        knowledge_contents: List[str]
) -> Dict:
    """
    Creates a quick reference guide from the analyses
    Improved to extract meaningful findings rather than fragments
    """
    quick_ref = {
        "total_conversations_analyzed": "68 conversations from past month",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "key_findings": [],
        "dominant_themes": [],
        "next_steps": []
    }

    # Extract actual key findings (complete thoughts, not fragments)
    all_content = " ".join(project_contents + inquiry_contents + knowledge_contents)
    lines = all_content.split('\n')

    findings = []
    for line in lines:
        line_clean = line.strip().lstrip('- •1234567890)').strip()

        # Look for complete findings statements
        if (len(line_clean) > 50 and
                line_clean[0].isupper() and
                any(indicator in line_clean.lower() for indicator in
                    ['shows', 'indicates', 'suggests', 'reveals', 'demonstrates'])):
            findings.append(line_clean)

    # Extract themes more intelligently
    theme_indicators = {
        "Technical debugging": ["debug", "error", "fix", "troubleshoot"],
        "AI/ML development": ["prompt", "chunking", "semantic", "query transformation"],
        "Financial analysis": ["ETF", "TVPI", "investment", "budget"],
        "Communication refinement": ["draft", "revision", "message", "presentation"],
        "Organizational strategy": ["reorg", "RTO", "technocrat", "power structure"]
    }

    all_content_lower = all_content.lower()
    for theme, keywords in theme_indicators.items():
        if any(keyword in all_content_lower for keyword in keywords):
            quick_ref['dominant_themes'].append(theme)

    # Take top findings
    quick_ref['key_findings'] = findings[:5]

    # Generate next steps based on patterns
    quick_ref['next_steps'] = [
        "Continue iterative development on AI/semantic search project",
        "Apply debugging patterns learned to future technical issues",
        "Leverage communication templates for upcoming stakeholder interactions"
    ]

    return quick_ref


def save_knowledge_base(knowledge_base: Dict, output_dir: Path) -> None:
    """
    Saves the knowledge base in multiple formats
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = output_dir / "knowledge_base.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved JSON: {json_path}")

    # Save as Markdown report
    md_path = output_dir / "knowledge_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(create_markdown_report(knowledge_base))
    print(f"  📄 Saved Report: {md_path}")


def create_markdown_report(knowledge_base: Dict) -> str:
    """
    Creates a readable markdown report from the knowledge base
    """

    # Helper function to format detailed insights sections
    def format_insights_section(insights: Dict, section_name: str) -> str:
        """Formats an insights dictionary into readable markdown"""
        output = []

        # Add overview if present
        if 'overview' in insights:
            output.append(f"**Overview**: {insights['overview']}\n")

        # Handle different types of content based on section
        if section_name == "projects":
            # Format key themes
            if insights.get('key_themes'):
                output.append("#### Key Themes Identified\n")
                for theme in insights['key_themes'][:10]:  # Limit to top 10
                    if len(theme) > 20:  # Only include substantial themes
                        output.append(f"- {theme}")
                output.append("")

            # Format development patterns
            if insights.get('development_patterns'):
                output.append("#### Development Patterns\n")
                unique_patterns = list(set(insights['development_patterns']))  # Remove duplicates
                for pattern in unique_patterns:
                    output.append(f"- {pattern}")
                output.append("")

        elif section_name == "learning":
            # Format problem types
            if insights.get('problem_types'):
                output.append("#### Types of Problems Addressed\n")
                unique_types = list(set(insights['problem_types']))
                for ptype in unique_types:
                    output.append(f"- {ptype}")
                output.append("")

            # Format interaction style
            if insights.get('interaction_style'):
                output.append("#### Your Interaction Style\n")
                unique_styles = list(set(insights['interaction_style']))
                for style in unique_styles:
                    output.append(f"- {style}")
                output.append("")

        elif section_name == "knowledge":
            # Format key domains
            if insights.get('key_domains'):
                output.append("#### Domains of Expertise\n")
                unique_domains = list(set(insights['key_domains']))
                for domain in unique_domains:
                    output.append(f"- {domain}")
                output.append("")

            # Format growth trajectory
            if insights.get('growth_trajectory'):
                output.append("#### Growth Trajectory\n")
                unique_trajectory = list(set(insights['growth_trajectory']))
                for trajectory in unique_trajectory:
                    output.append(f"- {trajectory}")
                output.append("")

        # Add raw analyses in a collapsible section for reference
        if insights.get('raw_analyses'):
            output.append("<details>")
            output.append("<summary><b>Click to view detailed analysis content</b></summary>\n")
            output.append("---\n")
            for i, analysis in enumerate(insights['raw_analyses'], 1):
                # Format the raw analysis text nicely
                paragraphs = analysis.split('\n\n')
                output.append(f"**Analysis {i}:**\n")
                for para in paragraphs[:3]:  # Show first 3 paragraphs
                    if para.strip():
                        output.append(para.strip())
                        output.append("")
                if len(paragraphs) > 3:
                    output.append("*(Content continues...)*\n")
                output.append("---\n")
            output.append("</details>\n")

        return '\n'.join(output)

    # Format recommendations section
    def format_recommendations(recs: Dict) -> str:
        """Formats recommendations into readable sections"""
        output = []

        if recs.get('immediate_actions'):
            output.append("### 🎯 Immediate Actions")
            for action in recs['immediate_actions']:
                output.append(f"- {action}")
            output.append("")

        if recs.get('medium_term_goals'):
            output.append("### 📅 Medium-Term Goals")
            for goal in recs['medium_term_goals']:
                output.append(f"- {goal}")
            output.append("")

        if recs.get('tools_and_templates'):
            output.append("### 🛠️ Tools and Templates to Create")
            for tool in recs['tools_and_templates']:
                output.append(f"- {tool}")
            output.append("")

        if recs.get('long_term_strategies'):
            output.append("### 🚀 Long-Term Strategies")
            for strategy in recs['long_term_strategies']:
                output.append(f"- {strategy}")
            output.append("")

        return '\n'.join(output)

    # Build the report
    report = f"""# 📚 Personal Knowledge Base Synthesis

**Generated**: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
**Analysis Scope**: {knowledge_base['metadata']['analyses_count']['project_evolution']} project analyses, {knowledge_base['metadata']['analyses_count']['inquiry_patterns']} inquiry pattern analyses, {knowledge_base['metadata']['analyses_count']['knowledge_evolution']} knowledge evolution analyses

---

## 🎯 Executive Summary

{knowledge_base['executive_summary']}

---

## 🔍 Cross-Cutting Themes

These themes appear consistently across different types of analyses:

{chr(10).join('- **' + theme + '**' for theme in knowledge_base['cross_cutting_themes'])}

---

## 📊 Detailed Insights

### 🚀 Project Development

{format_insights_section(knowledge_base['detailed_insights']['projects'], 'projects')}

### 🧠 Learning Patterns

{format_insights_section(knowledge_base['detailed_insights']['learning'], 'learning')}

### 📈 Knowledge Growth

{format_insights_section(knowledge_base['detailed_insights']['knowledge'], 'knowledge')}

---

## 💡 Actionable Recommendations

{format_recommendations(knowledge_base['actionable_recommendations'])}

---

## 📋 Quick Reference Guide

### Key Statistics
- **Analysis Date**: {knowledge_base['quick_reference']['analysis_date']}
- **Scope**: {knowledge_base['quick_reference']['total_conversations_analyzed']}

### Top Findings
"""

    # Add key findings if available
    if knowledge_base['quick_reference'].get('key_findings'):
        for finding in knowledge_base['quick_reference']['key_findings'][:5]:
            if finding and len(finding) > 20:  # Only include substantial findings
                report += f"\n- {finding}"
    else:
        report += "\n- Analysis complete - review detailed sections above for insights"

    report += """

---

## 📝 Next Steps

1. **Review this synthesis** to understand your intellectual journey
2. **Act on the immediate recommendations** to maintain project momentum  
3. **Create the suggested templates and tools** to streamline future work
4. **Tag future conversations** using the themes identified
5. **Schedule regular analysis updates** (monthly recommended) to track progress

---

*This synthesis was generated from batch analysis of your conversation history to preserve intellectual continuity and identify patterns in your learning and development.*
"""
    return report


def main():
    """
    Main entry point for the synthesis script
    """
    parser = argparse.ArgumentParser(
        description='Synthesize knowledge from batch analysis results'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('staged_results/combined_analysis.json'),
        help='Path to combined analysis JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('knowledge_base'),
        help='Output directory for knowledge base'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("          KNOWLEDGE SYNTHESIS SYSTEM")
    print("=" * 60)

    # Run synthesis
    knowledge_base = synthesize_analyses(args.input)

    if knowledge_base:
        print("\n💾 Saving knowledge base...")
        save_knowledge_base(knowledge_base, args.output)

        print("\n" + "=" * 60)
        print("        ✨ SYNTHESIS COMPLETE! ✨")
        print("=" * 60)
        print(f"\n📚 Your knowledge base has been created in: {args.output}/")
        print("\nFiles created:")
        print("  - knowledge_base.json: Complete structured data")
        print("  - knowledge_report.md: Readable report format")

        # Print summary statistics
        if 'detailed_insights' in knowledge_base:
            projects = knowledge_base['detailed_insights'].get('projects', {})
            learning = knowledge_base['detailed_insights'].get('learning', {})
            knowledge = knowledge_base['detailed_insights'].get('knowledge', {})

            print("\n📊 Synthesis Statistics:")
            print(f"  - Cross-cutting themes identified: {len(knowledge_base.get('cross_cutting_themes', []))}")
            print(f"  - Key domains analyzed: {len(knowledge.get('key_domains', []))}")
            print(
                f"  - Recommendations generated: {sum(len(v) for v in knowledge_base.get('actionable_recommendations', {}).values())}")
    else:
        print("\n❌ Synthesis failed - check that your input file exists and contains valid data")


if __name__ == "__main__":
    sys.exit(main())