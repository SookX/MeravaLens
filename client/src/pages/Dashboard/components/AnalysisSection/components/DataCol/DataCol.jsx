import { Grid } from "@mui/material"
import Location from "./components/Location/Location"
import Weather from "./components/Weather/Weather"
import Atmosphere from "./components/Atmosphere/Atmosphere"
import AirQuality from "./components/AirQuality/AirQuality"


const DataCol = () => {
    return (
        <Grid container spacing={2}>
            <Grid size={12}>
                <Location />
            </Grid>
            <Grid size={{ xs: 12, sm: 5, md: 12, lg: 5}}>
                <Weather />
            </Grid>
            <Grid size={{ xs: 12, sm: "grow", md: 12, lg: "grow" }}>
                <Atmosphere />
            </Grid>
            <Grid size={12}>
                <AirQuality />
            </Grid>
        </Grid>
    )
}

export default DataCol