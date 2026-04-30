from src.ingest.sector_flow.nsdl import extract_report_urls, parse_sector_flow_report


def test_extract_report_urls_dedupes_matches() -> None:
    html = """
    <html><body>
    <a href="~/StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/FIIInvestSector_Sep302025.html">A</a>
    <a href="~/StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/FIIInvestSector_Sep302025.html">B</a>
    </body></html>
    """
    result = extract_report_urls(html)
    assert result == [
        "https://pilot.fpi.nsdl.co.in/StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/FIIInvestSector_Sep302025.html"
    ]


def test_parse_sector_flow_report_uses_current_auc_and_latest_flow() -> None:
    html = """
    <table>
      <tr>
        <td></td><td></td>
        <td>AUC as on September 15, 2025</td>
        <td>Net Investment September 16-30, 2025</td>
        <td>AUC as on September 30, 2025</td>
      </tr>
      <tr>
        <td></td><td></td>
        <td>IN INR Cr.</td>
        <td>IN INR Cr.</td>
        <td>IN INR Cr.</td>
      </tr>
      <tr>
        <td></td><td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>Sr. No.</td><td>Sectors</td>
        <td>Total</td>
        <td>Total</td>
        <td>Total</td>
      </tr>
      <tr>
        <td>1</td><td>IT</td>
        <td>100</td>
        <td>15</td>
        <td>115</td>
      </tr>
    </table>
    """
    result = parse_sector_flow_report(html, "https://example.com/report.html")
    row = result.iloc[0]
    assert str(row["fortnight_end_date"].date()) == "2025-09-30"
    assert row["fpi_investment_value"] == 115
    assert row["fpi_change_abs"] == 15
    assert row["fpi_change_pct"] == 0.15
