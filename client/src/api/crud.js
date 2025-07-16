import axios from "axios";

axios.defaults.baseURL = 'http://127.0.0.1:8000/api'

// Makes a CRUD operation to the backend server
export const crud = async ({ url, method, body = null, headers = null }) => {
    const access = localStorage.getItem('access') || sessionStorage.getItem('access')

    try {
        const config = {
            headers: access ? {
                'Authorization': `Bearer ${access}`,
                ...headers
            } : {
                headers
            }
        }

        let response;
        if (method.toLowerCase() === 'get' || method.toLowerCase() === 'delete') {
            response = await axios[method](url, config);
        } else {
            response = await axios[method](url, body, config);
        }

        if(response) return response
    } catch(err) {
        return err
    }
}