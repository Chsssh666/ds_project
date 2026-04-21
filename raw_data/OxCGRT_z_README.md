![Data link to OxCGRT](https://github.com/OxCGRT/covid-policy-tracker/workflows/Data%20link%20to%20OxCGRT/badge.svg) <-- status of connection to OxCGRT database

Our data is published as CSV files here in the data folder of our OxCGRT/covid-policy-tracker repository. This is our recommended option to access our data.

The data is provided in several formats – including a simple [timeseries](/timeseries) format – and subnational data is contained in the specific folders. 

Note about subnational files:
- The `vaccines_full` file in the main folder _also_ contains subnational jurisdictions for the raw V1-V4 vaccine policy indicators (this is not also published in subnational folders). The [codebook](/documentation/codebook.md) and [index methodology](documentation/index_methodology.md) documentation contains more information about what each indicator and index represents.
- We are still in the process of collecting differentiated data for Brazilian subnational units, and so these files are published using our old legacy data format. (More information on our legacy data can be found in our [OxCGRT/covid-policy-tracker-legacy](https://github.com/OxCGRT/covid-policy-tracker-legacy) repo.)

The following table describes each type of file in turn. Below that, we have simpler tables that break down the files along specific dimensions (eg. which files contain which indicators, which files contain differentiated policies).   


| Csv file | Jurisdictions | Contains | Reports | Data user notes |
| --- | --- | --- | --- | --- |
| `latest` | All national jurisdictions in one file.  Separate folders by country for subnational data | A single variable for each indicator: <ul> <li>  For C1-C7, H6 and H8, this is the ‘majority’ (M) version </li><li>  For C8 this is C8EV </li><li>  For V1-V4 these are the summary indicators </li><li> A single `average` version of each index </ul> |Country/territory- and state-level data presented in "country/territory-day" format (or "state-day" as the case may be), with a list of all indicators for each country/territory as a single row each day |
| `differentiated_withnotes` <br> (split into 2020, 2021, and 2022 files) | All national  jurisdictions in one file. Separate folders by country for subnational data | All versions of C, E, and H indicators (eg. includes E, NV, N, and M versions where there is differentiated coding) <ul> <li> Summary variables for V1-V4  </li><li> Four versions of each index (with different weighting)  </li><li> Detailed notes explaining our policy coding  </ul>  | Reports country/territory- and state-level data in "country/territory-day" format with a column of notes from our data collectors for each indicator | This is the only csv file which contains notes. Split into years due to size |
|`timeseries` <br>(multiple CSVs)| All jurisdictions in one file | A single variable for each indicator: <ul> <li> For C1-C7, H6 and H8, this is the `majority` (M) version </li><li> For C8 this is C8EV </li><li> For V1-V4 these are the summary indicators </li><li> A single `average` version of each index </ul> | Country/territory-level data as individual timeseries for each indicator in CSV format, as well as a combined Excel file with a tab for each indicator | Each indicator and index is presented as a separate CSV or a separate worksheet in timeseries.xlsx. <br> Each CSV reports a single indicator arranged in time series format |
| `vaccines_full` | All jurisdictions in one file | All vaccine indicators | Country/region/territory data presented in "country/region/territory-day" | This is the only file that contains data for V1-V4 organised by all 53 categories, other files only contain the summary indicators|
| `latest_all_changes` | All national  jurisdictions in one file | A single variable for each indicator: <ul> <li>  For C1-C7, H6 and H8, this is the ‘majority’ (M) version </li><li> For C8 this is C8EV </li><li> For V1-V4 these are the summary indicators </li><li> A single `average` version of each index </ul>  | Reports country/territory-level data with a list of every change to the database. Every time a policy value changes, or every time a note is added to an indicator, it is represented with its own new row | All indicators, highlights changes in this order: Previous Policy - Current Policy - Current Note - Note used to change policy
| `latest_combined*` | All national  jurisdictions in one file.  Separate folders by country for subnational data | All national  jurisdictions in one file | A single variable for each indicator: <ul> <li>  For C1-C7, H6 and H8, this is the ‘majority’ (M) version </li><li>  For C8 this is C8EV </li><li>  For V1-V4 these are the summary indicators </li><li> A single `average` version of each index | Country/territory- and state-level data in "country/territory-day" format, but gives a single "combined" value for each indicator </ul>  | Latest with policies recorded as 1G, 2G etc instead of separate Policy (1, 2, 3 etc) and Flag (0/1) columns |
| `latest_responses` | All national  jurisdictions in one file. Separate folders by country for subnational data | A single variable for each indicator: <ul> <li>  For C1-C7, H6 and H8, this is the ‘majority’ (M) version </li><li>  For C8 this is C8EV </li><li> For V1-V4 these are the summary indicators </li><li> A single `average` version of each index </ul>  | | Duration of policy codes for a country along with initial notes. Sorted by PolicyType - PolicyCode - Start Date - End Date - Initial Note |

Prior to July 2022 we published data in a different structure wihtout this differentiated information. We continue to publish data under that old structure in our [legacy repo](https://github.com/OxCGRT/covid-policy-tracker-legacy).



### Coverage of specific indicators

We collect data across five different types of policy indicators (C, E, H, M, V). The data that relate to vaccine policy (eg. avaialbility of vaccines to different populations) is spread across over 100 raw variables (as [described in our codebook](/documentation/codebook.md#vaccination-policies)), and so some of our files contain 10 simple summary indicators of this information.

| File designation | C | E | H | M | Vaccine summary | Vaccine raw | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `latest` | yes | yes | yes | yes | yes | | |
| `differentiated_withnotes` | yes | yes | yes | yes | yes | | yes |
| `timeseries` | yes | yes | yes | yes | | |
| `vaccines_full` | | | | | yes | yes |
| `latest_all_changes` | yes | yes | yes | yes | yes | | yes |
| `latest_responses` | yes | yes | yes | yes | yes | |yes|
| `latest_combined*` | yes | yes | yes | yes | yes | | |

*Note: In the `latest_combined` files, please note that as described in the codebook, many of our indicators are recorded across two variables: one that records the strictness of the policy, and one that records its scope. 
- This is reported as a combination of the policy level (a number) and the scope flag (a letter: T for targeted policies or G for general policie; or F/A flags for indicator E1). For instance, for C3_Cancel public events we would have 0, 1T (recommend cancelling in some areas), 1G (recommend cancelling everywhere), 2T (require cancelling in some areas), 2G (require cancelling everywhere).
- We also include a numerical combination, using the same methodology to calculate compenents for our indices: a targeted policy is considered a half-step lower than a general jurisdiction-wide policy. For instance, for C3_Cancel public events we would have 0, 0.5 (recommend cancelling in some areas), 1(recommend cancelling everywhere), 1.5 (require cancelling in some areas), 2 (require cancelling everywhere).


### Data relating to differentiated policies

For some of our containment and health indicators (C and H), we differentiate the policies depending on whether they apply to everyone (`E`), non-vaccinated people (`NV`), or vaccinated people (`V`). This information, combined with the vaccination rate, allows us to determine what policy applies to the majority (`M`) of people. We publish different combinations of this information in different files. (Our [codebook](/documentation/codebook.md#differentiation-of-policies-by-vaccine-status) has more information about these differentiated policies.)

| File designation | E | NV | V | M* |
| --- | --- | --- | --- | --- |
| `latest` | | | | yes |
| `differentiated_withnotes` | yes | yes | yes | yes |
| `timeseries` | | | | yes |
| `latest_all_changes` | | | | yes |
| `latest_responses` | | | | yes |
| `latest_combined` | | | | yes |
