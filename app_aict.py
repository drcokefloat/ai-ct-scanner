import streamlit as st
import pandas as pd
import requests
import warnings
import urllib3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Suppress SSL warning
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

# Page config
st.set_page_config(
    page_title="AI Clinical Trials Scanner",
    page_icon="🤖",
    layout="wide"
)

st.title("Clinical Trials Monitor - AI in Health (CM/KL Prototype)")

# Add this after the title but before fetching data
st.markdown("""
## About This App
This dashboard monitors clinical trials involving artificial intelligence and machine learning technologies, including diagnostic AI, predictive models, generative AI, and other AI applications in healthcare.

The scanner maintains a live, rolling 12-month view of trials, automatically updating to show the most recent year of AI research activity on ClinicalTrials.gov.
""")

with st.expander("🔍 Search Methodology"):
    st.markdown("""
    ### Search Strategy
    
    **Primary Search Terms:**
    1. Core AI Terms:
       - "artificial intelligence", "machine learning"
       - "deep learning"
       - "neural network"
    
    2. Specialised AI Terms:
       - "natural language processing"
       - "computer vision"
       - "LLM", "GPT", "ChatGPT"
       - "generative AI"
    
    **Search Implementation:**
    - Using ClinicalTrials.gov API v2
    - Limited to 1000 most relevant trials (API limit)
    - Searching across all study fields
    - Date range: Last 12 months
    - No geographical restrictions
    """)

with st.expander("🏷️ Categorisation Methodology"):
    st.markdown("""
    ### How Trials are Categorised
    
    Trials are automatically categorised based on keywords in their titles and descriptions:
    
    **Diagnostic AI**
    - Keywords: diagnosis, detection, screening, imaging
    - Focus: Disease detection and medical imaging analysis
    
    **Predictive AI**
    - Keywords: prediction, prognosis, risk assessment, forecasting
    - Focus: Disease progression and outcome prediction
    
    **Robotics/Surgery AI**
    - Keywords: robot, surgical, surgery, intervention
    - Focus: Surgical assistance and robotic interventions
    
    **Drug Discovery AI**
    - Keywords: drug, molecule, compound, discovery
    - Focus: Drug development and molecular analysis
    
    **Monitoring AI**
    - Keywords: monitoring, tracking, surveillance
    - Focus: Patient monitoring and health tracking
    
    **Process Optimisation AI**
    - Keywords: optimisation, workflow, efficiency
    - Focus: Healthcare process improvements
    
    **Personalised Medicine AI**
    - Keywords: personalised, precision, treatment
    - Focus: Treatment customisation
    
    **Analysis/Classification AI**
    - Keywords: analysis, classification, pattern
    - Focus: General data analysis and pattern recognition
    
    **Generative AI Applications**
    - Language: chat, language, text, NLP
    - Image: visual, diffusion, image generation
    - Other: other generative applications
    
    **Note:** Categories are assigned based on primary focus, though trials may involve multiple AI applications.
    """)

# Core search function
@st.cache_data(ttl=3600)
def fetch_trials():
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # Calculate date 12 months ago
    twelve_months_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Search with rolling 12 month window
    params = {
        "format": "json",
        "query.term": (
            '("artificial intelligence" OR "machine learning") AND '
            f'AREA[StartDate]RANGE[{twelve_months_ago},MAX]'
        ),
        "pageSize": 1000  # API maximum
    }
    
    try:
        all_studies = []
        
        # Get first page
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        total_count = data.get('totalCount', 0)
        all_studies.extend(data.get('studies', []))
        
        # Keep fetching next pages while we have a next page token
        while 'nextPageToken' in data and len(all_studies) < total_count:
            params['pageToken'] = data['nextPageToken']
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            all_studies.extend(data.get('studies', []))
            st.write(f"Retrieved {len(all_studies)} of {total_count} trials...")
        
        processed_studies = []
        for study in all_studies:
            protocol = study.get('protocolSection', {})
            processed_study = {
                'nct_id': protocol.get('identificationModule', {}).get('nctId'),
                'title': protocol.get('identificationModule', {}).get('briefTitle'),
                'status': protocol.get('statusModule', {}).get('overallStatus'),
                'phase': '; '.join(protocol.get('designModule', {}).get('phases', []) or ['N/A']),
                'start_date': protocol.get('statusModule', {}).get('startDateStruct', {}).get('date'),
                'sponsor': protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name'),
                'intervention_type': '; '.join([i.get('type') for i in protocol.get('armsInterventionsModule', {}).get('interventions', []) if i.get('type')]),
                'intervention_names': '; '.join([i.get('name') for i in protocol.get('armsInterventionsModule', {}).get('interventions', []) if i.get('name')]),
                'locations': '; '.join([
                    loc.get('country', '') for loc in protocol.get('contactsLocationsModule', {}).get('locations', []) if loc.get('country')
                ])
            }
            
            # Enhanced AI categorisation with more specific categories
            title_desc = (processed_study['title'] or '').lower()
            desc = (protocol.get('descriptionModule', {}).get('briefSummary', '') or '').lower()
            full_text = title_desc + ' ' + desc

            # Generative AI indicators
            gen_ai_terms = ['generative ai', 'gpt', 'llm', 'large language model', 'chatgpt', 
                          'stable diffusion', 'dall-e', 'text-to-image', 'foundation model']
            
            # Categorise AI type with more specific categories
            if any(term in full_text for term in gen_ai_terms):
                if any(term in full_text for term in ['chat', 'language', 'text', 'nlp']):
                    processed_study['ai_type'] = 'Generative AI - Language'
                elif any(term in full_text for term in ['image', 'visual', 'diffusion']):
                    processed_study['ai_type'] = 'Generative AI - Image'
                else:
                    processed_study['ai_type'] = 'Generative AI - Other'
            elif any(term in full_text for term in ['diagnosis', 'detection', 'screening', 'imaging']):
                processed_study['ai_type'] = 'Diagnostic AI'
            elif any(term in full_text for term in ['prediction', 'prognosis', 'risk assessment', 'forecasting']):
                processed_study['ai_type'] = 'Predictive AI'
            elif any(term in full_text for term in ['robot', 'surgical', 'surgery', 'intervention']):
                processed_study['ai_type'] = 'Robotics/Surgery AI'
            elif any(term in full_text for term in ['drug', 'molecule', 'compound', 'discovery']):
                processed_study['ai_type'] = 'Drug Discovery AI'
            elif any(term in full_text for term in ['monitoring', 'tracking', 'surveillance']):
                processed_study['ai_type'] = 'Monitoring AI'
            elif any(term in full_text for term in ['optimization', 'workflow', 'efficiency']):
                processed_study['ai_type'] = 'Process Optimisation AI'
            elif any(term in full_text for term in ['personalized', 'precision', 'treatment']):
                processed_study['ai_type'] = 'Personalized Medicine AI'
            elif any(term in full_text for term in ['analysis', 'classification', 'pattern']):
                processed_study['ai_type'] = 'Analysis/Classification AI'
            else:
                processed_study['ai_type'] = 'Other AI Applications'
            
            # Add UK involvement
            locations = protocol.get('contactsLocationsModule', {}).get('locations', [])
            countries = [loc.get('country') for loc in locations if loc.get('country')]
            processed_study['uk_involvement'] = 'United Kingdom' in countries
            
            processed_studies.append(processed_study)
            
        return pd.DataFrame(processed_studies)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# Fetch data
with st.spinner('Scanning clinical trials database...'):
    trials_df = fetch_trials()

if not trials_df.empty:
    st.markdown("## AI in Clinical Trials")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total AI Trials", len(trials_df))
        
    with col2:
        active_trials = len(trials_df[trials_df['status'] == 'RECRUITING'])
        st.metric("Currently Recruiting", active_trials)
        
    with col3:
        types_count = trials_df['ai_type'].value_counts()
        most_common = types_count.index[0] if not types_count.empty else "None"
        st.metric("Most Common Application", most_common)

    # After initial metrics
    st.markdown("---")  # Add divider
    
    st.markdown("### Key Analytics")
    st.markdown("")  # Add space

    # Create two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        # AI Applications Distribution - Clean Pie Chart
        fig_types = px.pie(
            trials_df,
            names='ai_type',
            title='Global Distribution of AI Applications',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_types.update_traces(
            textposition='inside',  # Move labels inside
            textinfo='percent'  # Show only percentages inside
        )
        fig_types.update_layout(
            showlegend=True,  # Show legend instead
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.0
            ),
            title_x=0.5,
            margin=dict(t=50, l=20, r=120, b=20),  # Adjusted margins for legend
            height=400
        )
        st.plotly_chart(fig_types, use_container_width=True)

    with col2:
        # Trial Status Breakdown - Clear Bar Chart
        status_counts = trials_df['status'].value_counts()
        fig_status = go.Figure(data=[
            go.Bar(
                x=status_counts.values,
                y=status_counts.index,
                orientation='h',
                marker_color='rgb(158,202,225)'
            )
        ])
        fig_status.update_layout(
            title='Global Trial Status Overview',
            title_x=0.5,
            xaxis_title='Number of Trials',
            margin=dict(t=50, l=20, r=20, b=20),
            height=400
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # After existing charts
    st.markdown("")
    st.markdown("### Global Leadership in AI Trials")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Top Countries Bar Chart - with data cleaning and proper counting
        country_counts = pd.Series([
            country.strip()
            for locations in trials_df['locations'].dropna()
            for country in set(locations.split(';'))  # Use set() to count each country only once per trial
            if country.strip()
        ]).value_counts().head(10)
        
        # Remove any empty strings or whitespace-only entries
        country_counts = country_counts[country_counts.index.str.len() > 0]
        
        fig_countries = go.Figure(data=[
            go.Bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                marker_color='rgb(102,194,165)'
            )
        ])
        fig_countries.update_layout(
            title={
                'text': 'Top 10 Countries Leading AI Trials<br><sup>Each trial counted once per country</sup>',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Number of Trials',
            margin=dict(t=50, l=20, r=20, b=20),
            height=400
        )
        st.plotly_chart(fig_countries, use_container_width=True)

    with col4:
        # Top Sponsors Bar Chart
        sponsor_counts = trials_df['sponsor'].value_counts().head(10)
        fig_sponsors = go.Figure(data=[
            go.Bar(
                x=sponsor_counts.values,
                y=sponsor_counts.index,
                orientation='h',
                marker_color='rgb(252,141,98)'
            )
        ])
        fig_sponsors.update_layout(
            title='Top 10 Organizations Leading AI Trials',
            title_x=0.5,
            xaxis_title='Number of Trials',
            margin=dict(t=50, l=20, r=20, b=20),
            height=400
        )
        st.plotly_chart(fig_sponsors, use_container_width=True)

    # After the Global Leadership section and before the database section
    st.markdown("---")  # Add divider
    st.markdown("### 📈 AI Innovation Trends")
    
    col5, col6 = st.columns(2)

    with col5:
        # Monthly Growth Trend with proper datetime handling
        trials_df['month_date'] = pd.to_datetime(trials_df['start_date'], format='mixed')
        twelve_months_ago = datetime.now() - timedelta(days=365)
        current_date = datetime.now()
        
        # Filter using datetime objects - exclude future dates
        monthly_trend = trials_df[
            (trials_df['month_date'] >= twelve_months_ago) & 
            (trials_df['month_date'] <= current_date)
        ]
        monthly_trend['month'] = monthly_trend['month_date'].dt.strftime('%Y-%m')
        monthly_trend = monthly_trend.groupby('month').size().reset_index()
        monthly_trend = monthly_trend.sort_values('month')
        
        # Monthly Growth Trend chart
        fig_trend = px.line(monthly_trend, x='month', y=0,
                           title='Monthly AI Trial Initiation Trend (Last 12 Months)',
                           labels={'month': 'Month', '0': 'New Trials'},
                           markers=True)
        fig_trend.update_layout(
            title={
                'text': 'Monthly AI Trial Initiation Trend (Last 12 Months)',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(t=50, l=20, r=20, b=20),
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col6:
        # AI Type Evolution with proper datetime filtering
        filtered_trials = trials_df[
            (trials_df['month_date'] >= twelve_months_ago) & 
            (trials_df['month_date'] <= current_date)
        ]
        filtered_trials['month'] = filtered_trials['month_date'].dt.strftime('%Y-%m')
        
        type_by_month = pd.crosstab(
            filtered_trials['month'],
            filtered_trials['ai_type']
        ).sort_index()
        
        # AI Type Evolution chart
        fig_evolution = px.area(type_by_month, 
                              title='Evolution of AI Applications (Last 12 Months)')
        fig_evolution.update_layout(
            title={
                'text': 'Evolution of AI Applications (Last 12 Months)',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Month',
            yaxis_title='Number of Trials',
            showlegend=True,
            margin=dict(t=50, l=20, r=20, b=20),
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_evolution, use_container_width=True)

    # After AI Innovation Trends section
    st.markdown("---")
    st.markdown("### 🌐 Global Research Network")
    
    # Create collaboration network based on countries in same trials
    collaborations = []
    for _, trial in trials_df.iterrows():
        # Get unique countries in this trial
        countries = list(set([c.strip() for c in str(trial['locations']).split(';') if c.strip()]))
        if len(countries) > 1:  # Only consider multi-country trials
            for i in range(len(countries)):
                for j in range(i+1, len(countries)):
                    if countries[i] != countries[j]:  # Only add if countries are different
                        collaborations.append((countries[i], countries[j]))
    
    # Create and populate network
    G = nx.Graph()
    for c1, c2 in collaborations:
        if G.has_edge(c1, c2):
            G[c1][c2]['weight'] += 1
        else:
            G.add_edge(c1, c2, weight=1)
    
    # Calculate node sizes based on number of trials
    node_sizes = {node: G.degree(node, weight='weight') * 20 for node in G.nodes()}
    
    # Create network layout with better spacing
    pos = nx.spring_layout(G, k=2, iterations=100)  # Increased k for more spacing
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create node traces
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Create edges trace with better visibility
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(169,169,169,0.5)'),  # Lighter, semi-transparent lines
        hoverinfo='none',
        mode='lines')
    
    # Create nodes trace with better labels
    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"{node}" for node in G.nodes()],
        textposition="top center",
        hovertext=[f"{node}<br>{G.degree(node)} collaborations" for node in G.nodes()],
        marker=dict(
            showscale=True,
            colorscale='Viridis',  # Changed colorscale
            size=[G.degree(node) * 10 for node in G.nodes()],  # Size based on connections
            color=[G.degree(node) for node in G.nodes()],
            line_width=1,
            colorbar=dict(
                title='Number of<br>Collaborations',
                thickness=15,
                x=1.02
            )
        ))

    # Create the figure with better layout
    fig_network = go.Figure(data=[edges_trace, nodes_trace],
                          layout=go.Layout(
                              title={
                                  'text': 'Global AI Trial Collaboration Network<br><sup>Node size and color indicate number of international collaborations</sup>',
                                  'x': 0.5,
                                  'xanchor': 'center',
                                  'yanchor': 'top'
                              },
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=80,t=60),  # Adjusted margins
                              height=600,
                              plot_bgcolor='white',  # White background
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )
    
    st.plotly_chart(fig_network, use_container_width=True)
    
    # Add some insights
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("#### Top Collaborative Countries")
        top_collaborators = sorted(
            [(node, G.degree(node)) for node in G.nodes()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for country, degree in top_collaborators:
            st.markdown(f"- **{country}**: {degree} collaborations")
            
    with col8:
        st.markdown("#### Strongest International Partnerships")
        strong_partnerships = sorted(
            [(u, v, d['weight']) for (u, v, d) in G.edges(data=True)
             if u != v],  # Ensure we're not counting same-country pairs
            key=lambda x: x[2],
            reverse=True
        )[:5]
        for c1, c2, weight in strong_partnerships:
            st.markdown(f"- **{c1} - {c2}**: {weight} collaborative trials")

    st.markdown("---")  # Add divider before database

    # -------------------------
    # Detailed Trial Database
    # -------------------------
    st.markdown("## 📋 Detailed Trial Database")
    st.markdown("Explore and filter all AI trials")  # Updated text to reflect all AI trials
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Filter by Status", 
            options=sorted(trials_df['status'].unique()),
            help="Select one or more trial statuses"
        )
    with col2:
        type_filter = st.multiselect(
            "Filter by AI Type",
            options=sorted(trials_df['ai_type'].unique()),
            help="Select one or more AI applications"
        )
    with col3:
        uk_only = st.checkbox("UK Trials Only", help="Show only trials with UK involvement")

    # Apply filters
    filtered_df = trials_df.copy()
    if status_filter:
        filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
    if type_filter:
        filtered_df = filtered_df[filtered_df['ai_type'].isin(type_filter)]
    if uk_only:
        filtered_df = filtered_df[filtered_df['uk_involvement']]

    # Show active filters
    active_filters = []
    if status_filter:
        active_filters.append(f"Status: {', '.join(status_filter)}")
    if type_filter:
        active_filters.append(f"AI Type: {', '.join(type_filter)}")
    if uk_only:
        active_filters.append("UK trials only")
        
    if active_filters:
        st.markdown("*Showing trials matching:* " + " | ".join(active_filters))
    else:
        st.markdown("*Showing all trials*")

    # Create clickable NCT links
    display_df = filtered_df.copy()
    display_df['NCT ID'] = display_df['nct_id'].apply(
        lambda x: f"https://clinicaltrials.gov/ct2/show/{x}"
    )
    
    # Display the database
    st.markdown("Click on any NCT ID to view the full trial record on ClinicalTrials.gov")
    st.dataframe(
        display_df[[
            'NCT ID', 'title', 'ai_type', 'status', 
            'phase', 'sponsor', 'start_date'
        ]],
        use_container_width=True,
        column_config={
            "NCT ID": st.column_config.LinkColumn("NCT ID", help="Click to view full trial record")
        }
    )

    # Show count of filtered results
    st.markdown(f"*Showing {len(filtered_df)} of {len(trials_df)} total trials*")

    # Download option
    st.download_button(
        "Download Full Dataset",
        trials_df.to_csv(index=False),
        "ai_trials.csv",
        "text/csv"
    )
else:
    st.error("No data available. Please try again later.")