import { Card, Stack, Typography } from "@mui/material"
import { useContext } from "react"
import { DashboardContext } from "../../../../../../Dashboard"

const Location = () => {
    // Gets dashboard data
    const { coords, weather } = useContext(DashboardContext)



    return (
        <Card sx={{ padding: 3 }}>
            <Typography color="primary" variant="h3">{weather.name}</Typography>
            <Stack direction="row" mb={1} gap={1} alignItems="center">
                <Typography variant="h5">Country: </Typography>
                <Typography variant="h5">{weather.sys.country}</Typography>
            </Stack>
            <Stack>
                <Typography variant="body1" color="primary.dark">Location: </Typography>
                <Typography variant="body1">({coords.lat}, {coords.lng})</Typography>
            </Stack>
        </Card>
    )
}

export default Location